from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from src.slab_edge_opening import extract_slab_edge_openings


DEFAULT_CONFIG = {
    'minimum_area_image_fraction': .001,
    'minimum_evidence_pixels': 64,
    'minimum_recession_fraction': .006,
    'minimum_opening_context_fraction': .025,
    'minimum_edge_context_fraction': .15,
    'minimum_partial_enclosure_fraction': .035,
    'minimum_ring_support_fraction': .15,
    'ring_radius_fraction': .008,
    'context_completion_radius_fraction': .12,
    'candidate_closing_fraction': .002,
    'maximum_inferred_internal_fill_fraction': .20,
    'minimum_geometry_coverage': .50,
    'maximum_facade_fraction_without_context': .25,
    'recession_distance_multiplier': 2.,
    'min_ray_cosine': .08,
    'edge_requires_opening_context': True,
    'occluder_minimum_angle_degrees': 60.,
    'occluder_normal_coherence': .94,
    'occluder_maximum_local_depth_range_fraction': .08,
}


def validated_config(config: Mapping[str,Any] | None = None) -> dict[str,Any]:
    result=dict(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config,Mapping):
            raise ValueError('config must be a mapping')
        unknown=set(config)-set(result)
        if unknown:
            raise ValueError('Unknown slab_holes settings: '+str(sorted(unknown)))
        result.update(config)
    for key,value in result.items():
        if key=='edge_requires_opening_context':
            if not isinstance(value,(bool,np.bool_)):
                raise ValueError(key+' must be boolean')
            continue
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)):
            raise ValueError(key+' must be numeric')
        value=float(value)
        if not np.isfinite(value) or value<0:
            raise ValueError(key+' must be finite and nonnegative')
        if 'fraction' in key or key=='min_ray_cosine':
            if value>1:
                raise ValueError(key+' must be in [0,1]')
        if key=='occluder_normal_coherence' and value>1:
            raise ValueError(key+' must be in [0,1]')
        if key=='occluder_minimum_angle_degrees' and value>90:
            raise ValueError(key+' must be in [0,90]')
        if key=='minimum_evidence_pixels':
            if value<1 or int(value)!=value:
                raise ValueError(key+' must be a positive integer')
            value=int(value)
        result[key]=value
    if result['min_ray_cosine']<=0 or result['recession_distance_multiplier']<=0:
        raise ValueError('Ray cosine and recession multiplier must be positive')
    return result


def _kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)


def _mask(value,shape,name):
    value=np.asarray(value)
    if value.dtype!=bool or value.shape!=shape:
        raise ValueError(name+' must be a boolean H x W array')
    return value


def _context(classes,labels,shape):
    masks={key:np.zeros(shape,bool) for key in ('opening','facade','wall','column')}
    resolved={key:[] for key in masks}
    if (classes is None)!=(labels is None):
        raise ValueError('semantic_classes and semantic_labels must be supplied together')
    if classes is None:
        return masks,resolved
    classes=np.asarray(classes)
    if classes.shape!=shape or not np.issubdtype(classes.dtype,np.integer) or not isinstance(labels,Mapping):
        raise ValueError('Full classes must be integer H x W; labels must map ids to names')
    names={'opening':{'railing','fence','bannister','banister','escalator','stairs','stairway'},
           'facade':{'door','poster','signboard','mirror','screen door','shopfront'},
           'wall':{'wall','building'},'column':{'column'}}
    for class_id,name in labels.items():
        if not isinstance(name,str):
            raise ValueError('Label names must be strings')
        name=name.strip().casefold()
        for key,expected in names.items():
            if name in expected:
                resolved[key].append(int(class_id))
    for key,ids in resolved.items():
        masks[key]=np.isin(classes,ids)
    return masks,resolved


def _directional_support(support):
    height,width=support.shape
    horizontal=cv2.morphologyEx(support.astype('uint8'),cv2.MORPH_OPEN,
        np.ones((1,max(3,int(round(width*.004)))),np.uint8)).astype(bool)
    vertical=cv2.morphologyEx(support.astype('uint8'),cv2.MORPH_OPEN,
        np.ones((max(3,int(round(height*.004))),1),np.uint8)).astype(bool)
    left=np.maximum.accumulate(horizontal,axis=1)
    right=np.maximum.accumulate(horizontal[:,::-1],axis=1)[:,::-1]
    above=np.maximum.accumulate(vertical,axis=0)
    below=np.maximum.accumulate(vertical[::-1],axis=0)[::-1]
    return left,right,above,below


def _plane_aligned_enclosure(support,intrinsics,plane):
    coefficients=plane.get('plane',plane)
    normal=np.asarray(coefficients['normal'],dtype=float)
    d=float(coefficients['d'])
    projected=-np.sign(d)*(np.linalg.inv(np.asarray(intrinsics,dtype=float)).T@normal)
    height,width=support.shape
    projected[:2]/=np.array([width,height],dtype=float)
    if np.linalg.norm(projected[:2])<1e-10:
        left,right,above,below=_directional_support(support)
        return left&right&above&below
    angle=-float(np.degrees(np.arctan2(projected[0],projected[1])))
    matrix=cv2.getRotationMatrix2D(((width-1)/2.,(height-1)/2.),angle,1.)
    cosine,sine=abs(matrix[0,0]),abs(matrix[0,1])
    new_width=int(np.ceil(width*cosine+height*sine))
    new_height=int(np.ceil(height*cosine+width*sine))
    matrix[0,2]+=(new_width-width)/2.
    matrix[1,2]+=(new_height-height)/2.
    aligned=cv2.warpAffine(support.astype('uint8'),matrix,(new_width,new_height),flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    left,right,above,below=_directional_support(aligned)
    bounded=left&right&below
    inverse=cv2.invertAffineTransform(matrix)
    return cv2.warpAffine(bounded.astype('uint8'),inverse,(width,height),flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)


def _geometry_for_verified_support(semantic,points,normals,valid,intrinsics,plane,shape,cfg):
    _mask(semantic,shape,'semantic');valid=_mask(valid,shape,'valid')
    points=np.asarray(points,dtype=float);normals=np.asarray(normals,dtype=float)
    if points.shape!=(*shape,3) or normals.shape!=points.shape:
        raise ValueError('points and normals must be matching H x W x 3 arrays')
    k=np.asarray(intrinsics,dtype=float)
    if k.shape!=(3,3) or not np.isfinite(k).all() or abs(np.linalg.det(k))<1e-12:
        raise ValueError('intrinsics must be finite invertible normalized 3 x 3')
    coefficients=plane.get('plane',plane)
    n=np.asarray(coefficients.get('normal'),dtype=float);d=float(coefficients.get('d',np.nan))
    threshold=float(plane.get('metadata',{}).get('effective_plane_distance_threshold',np.nan))
    if n.shape!=(3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n),1.) or not np.isfinite(d) or abs(d)<1e-10:
        raise ValueError('plane must have finite unit normal and nonzero camera offset')
    if not np.isfinite(threshold) or threshold<=0:
        raise ValueError('positive effective plane-distance tolerance is required')
    valid=valid&np.isfinite(points).all(axis=-1)&(points[...,2]>0)
    away=-np.sign(d)*(points@n+d);away[~valid]=np.nan
    lengths=np.linalg.norm(normals,axis=-1)
    normal_valid=valid&np.isfinite(normals).all(axis=-1)&(lengths>1e-8)
    angles=np.full(shape,np.nan)
    angles[normal_valid]=np.degrees(np.arccos(np.clip(np.abs(normals[normal_valid]@n/lengths[normal_valid]),0,1)))
    y,x=np.indices(shape,dtype=float)
    rays=np.stack([(x+.5)/shape[1],(y+.5)/shape[0],np.ones(shape)],axis=-1)@np.linalg.inv(k).T
    denominator=rays@n
    cosine=np.abs(denominator)/np.linalg.norm(rays,axis=-1)
    with np.errstate(divide='ignore',invalid='ignore'):
        scale=-d/denominator
    positive=np.isfinite(scale)&(scale>0)
    return {'normal_angle':angles,'signed_away':away,'ray_domain':positive&(cosine>=cfg['min_ray_cosine'])}


def detect_slab_holes(semantic,points,normals,valid,intrinsics,plane,config=None,*,
                         semantic_classes=None,semantic_labels=None,
                         surface_role='floor',verified_support=None):
    cfg=validated_config(config)
    if surface_role not in ('floor','ceiling'):
        raise ValueError('surface_role must be floor or ceiling')
    points=np.asarray(points)
    if points.ndim!=3 or points.shape[-1]!=3:
        raise ValueError('points must be H x W x 3')
    shape=points.shape[:2]
    if verified_support is None:
        base=extract_slab_edge_openings(semantic,points,normals,valid,intrinsics,plane,
            {'dominant_gap_only':False,'min_ray_cosine':cfg['min_ray_cosine'],
             'recess_distance_multiplier':cfg['recession_distance_multiplier']},
            semantic_classes=semantic_classes,semantic_labels=semantic_labels)
        support=base['recovered_slab_support'].copy()
    else:
        support=_mask(verified_support,shape,'verified_support').copy()
        base=_geometry_for_verified_support(semantic,points,normals,valid,intrinsics,plane,shape,cfg)
    if not support.any():
        raise ValueError('No verified current-slab support')
    valid_geometry=np.asarray(valid)&np.isfinite(points).all(axis=-1)&(np.asarray(points)[...,2]>0)
    contexts,resolved=_context(semantic_classes,semantic_labels,shape)
    threshold=float(plane.get('metadata',{}).get('effective_plane_distance_threshold'))
    recess=valid_geometry&(base['signed_away']>threshold*cfg['recession_distance_multiplier'])
    foreground=valid_geometry&(base['signed_away']<-threshold*cfg['recession_distance_multiplier'])
    parallel=np.isfinite(base['normal_angle'])&(base['normal_angle']<=25.)
    vectors=np.asarray(normals,dtype=float)
    lengths=np.linalg.norm(vectors,axis=-1)
    usable=valid_geometry&np.isfinite(vectors).all(axis=-1)&(lengths>1e-8)
    unit=np.zeros_like(vectors)
    unit[usable]=vectors[usable]/lengths[usable,None]
    window=max(3,2*int(round(min(shape)*.003))+1)
    mean=cv2.blur(unit.astype(np.float32),(window,window))
    coherence=np.linalg.norm(mean,axis=-1)
    depth=np.asarray(points)[...,2].astype(np.float32)
    depth_clean=np.where(valid_geometry,depth,0.)
    kernel=np.ones((window,window),np.uint8)
    depth_range=cv2.dilate(depth_clean,kernel)-cv2.erode(depth_clean,kernel)
    continuity=depth_range/np.maximum(abs(depth_clean),1e-8)
    explicit_occluder=(contexts['facade']|contexts['column'])&foreground&usable
    explicit_occluder&=(base['normal_angle']>=cfg['occluder_minimum_angle_degrees'])
    explicit_occluder&=(coherence>=cfg['occluder_normal_coherence'])
    explicit_occluder&=(continuity<=cfg['occluder_maximum_local_depth_range_fraction'])
    gaps=base['ray_domain']&~support
    enclosure=gaps&_plane_aligned_enclosure(support,intrinsics,plane)
    radius=max(1,int(round(min(shape)*cfg['candidate_closing_fraction'])))
    enclosed=cv2.morphologyEx(enclosure.astype('uint8'),cv2.MORPH_CLOSE,_kernel(radius)).astype(bool)&gaps
    minimum=max(1,int(np.ceil(np.prod(shape)*cfg['minimum_area_image_fraction'])))
    ring_radius=max(1,int(round(min(shape)*cfg['ring_radius_fraction'])))
    completion_radius=max(1,int(round(min(shape)*cfg['context_completion_radius_fraction'])))
    accepted=np.zeros(shape,bool)
    uncertain=np.zeros(shape,bool)
    bounded_uncertain=np.zeros(shape,bool)
    rejected=np.zeros(shape,bool)
    inferred_fill=np.zeros(shape,bool)
    records=[]
    accepted_record_ids=np.zeros(shape,np.int32)
    quality=plane.get('metadata',{}).get('quality_passed')

    def assess(component,kind,raw_id,extra=None):
        area=int(component.sum())
        if area<minimum:
            return
        ring=cv2.dilate(component.astype('uint8'),_kernel(ring_radius)).astype(bool)&~component
        ring_fraction=float(np.count_nonzero(ring&support)/max(1,int(ring.sum())))
        geo_count=int(np.count_nonzero(component&valid_geometry))
        recess_count=int(np.count_nonzero(component&recess))
        context_count=int(np.count_nonzero(component&contexts['opening']))
        context_fraction=context_count/area
        recession_fraction=recess_count/max(1,geo_count)
        enclosure_fraction=float(np.count_nonzero(component&enclosure)/area)
        facade_fraction=float(np.count_nonzero(component&(contexts['facade']|contexts['column']))/area)
        coverage=geo_count/area
        reasons=[]
        has_recession=(recess_count>=cfg['minimum_evidence_pixels'] and recession_fraction>=cfg['minimum_recession_fraction'])
        context_requirement=cfg['minimum_edge_context_fraction'] if kind=='edge' else cfg['minimum_opening_context_fraction']
        has_context=(context_count>=cfg['minimum_evidence_pixels'] and context_fraction>=context_requirement)
        if ring_fraction<cfg['minimum_ring_support_fraction']:
            reasons.append('insufficient_current_slab_boundary_support')
        if coverage<cfg['minimum_geometry_coverage']:
            reasons.append('insufficient_predicted_geometry_coverage')
        if not(has_recession or has_context):
            reasons.append('no_independent_recession_or_opening_context')
        bound_fraction=max(enclosure_fraction,(extra or {}).get('whole_raw_gap_partial_enclosure_fraction',0.))
        if kind=='edge' and bound_fraction<cfg['minimum_partial_enclosure_fraction'] and not has_context:
            reasons.append('unbounded_recession_could_be_wall_or_reflection')
        if kind=='edge' and cfg['edge_requires_opening_context'] and not has_context:
            reasons.append('external_edge_requires_independent_opening_context')
        if kind=='unsupported_external':
            reasons.append('external_gap_without_bounded_completion')
        if facade_fraction>cfg['maximum_facade_fraction_without_context'] and not has_context:
            reasons.append('facade_or_column_dominant_without_opening_context')
        if quality is False:
            reasons.append('target_plane_quality_failed')
        definite_rejection='facade_or_column_dominant_without_opening_context' in reasons
        if not reasons:
            accepted[:]|=component
            accepted_record_ids[component]=len(records)+1
            status='context_supported_opening_candidate' if has_context else 'recession_supported_opening_candidate'
        elif definite_rejection:
            rejected[:]|=component
            status='rejected_facade_candidate'
        else:
            uncertain[:]|=component
            if kind in ('enclosed','edge'):
                bounded_uncertain[:]|=component
            status='uncertain_gap'
        y,x=np.where(component)
        record={'id':len(records)+1,'raw_gap_id':raw_id,'kind':kind,'status':status,
            'area_pixels':area,'bbox_xywh':[int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)],
            'ring_current_slab_support_fraction':ring_fraction,'enclosure_fraction':enclosure_fraction,
            'geometry_coverage':coverage,'positive_recession_pixels':recess_count,
            'positive_recession_fraction':recession_fraction,'opening_context_fraction':context_fraction,
            'facade_column_fraction':facade_fraction,'wall_building_fraction':float(np.count_nonzero(component&contexts['wall'])/area),
            'foreground_or_glass_fraction':float(np.count_nonzero(component&foreground)/max(1,geo_count)),
            'reasons':reasons,'geometrically_confirmed_through_void':False}
        if extra:
            record.update(extra)
        records.append(record)

    nc,labels,stats,_=cv2.connectedComponentsWithStats(enclosed.astype('uint8'),8)
    for index in range(1,nc):
        if stats[index,cv2.CC_STAT_AREA]<minimum:
            continue
        component=labels==index
        filled=binary_fill_holes(component)&gaps
        added=filled&~component
        if added.sum()/max(1,int(component.sum()))<=cfg['maximum_inferred_internal_fill_fraction']:
            component=filled
            inferred_fill|=added
        assess(component,'enclosed',None)

    ng,labels,stats,_=cv2.connectedComponentsWithStats(gaps.astype('uint8'),8)
    for index in range(1,ng):
        if stats[index,cv2.CC_STAT_AREA]<minimum:
            continue
        component=labels==index
        remaining=component&~enclosed
        if remaining.sum()<minimum:
            continue
        context_seeds=component&contexts['opening']
        recess_seeds=component&recess&parallel
        partial_enclosure=float(np.count_nonzero(component&enclosure)/int(component.sum()))
        seeds=context_seeds.copy()
        if partial_enclosure>=cfg['minimum_partial_enclosure_fraction']:
            seeds|=recess_seeds
        evidence_count=int(seeds.sum())
        if evidence_count>=cfg['minimum_evidence_pixels']:
            distance=distance_transform_edt(~seeds)
            body=remaining&(distance<=completion_radius)
            body_nc,body_labels,body_stats,_=cv2.connectedComponentsWithStats(body.astype('uint8'),8)
            for body_id in range(1,body_nc):
                if body_stats[body_id,cv2.CC_STAT_AREA]>=minimum:
                    assess(body_labels==body_id,'edge',index,{'bounded_completion_radius_pixels':completion_radius,
                        'whole_raw_gap_partial_enclosure_fraction':partial_enclosure})
        leftover=component&~accepted&~uncertain&~rejected&~enclosed
        if leftover.sum()>=minimum:
            assess(leftover,'unsupported_external',index)
    uncertain&=~accepted
    rejected&=~accepted&~uncertain
    pre_filter_accepted=accepted.copy()
    excluded_occluders=accepted&explicit_occluder
    accepted&=~explicit_occluder
    for record in records:
        if record['status'].endswith('opening_candidate'):
            identity=accepted_record_ids==record['id']
            record['pre_occluder_filter_area_pixels']=record['area_pixels']
            record['excluded_predicted_solid_occluder_pixels']=int(np.count_nonzero(identity&excluded_occluders))
            record['area_pixels']=int(np.count_nonzero(identity&accepted))
            if record['area_pixels']==0:
                record['status']='rejected_predicted_solid_occluder'
    rejected|=excluded_occluders
    bounded_uncertain&=~accepted&~rejected
    observed=accepted&recess
    return {'opening_hypothesis_mask':accepted,'ambiguous_opening_mask':uncertain,
        'bounded_uncertain_mask':bounded_uncertain,
        'rejected_opening_mask':rejected,'recovered_slab_support':support,
        'raw_gap_mask':gaps,'enclosed_gap_mask':enclosed,'observed_recess_mask':observed,
        'inferred_void_mask':accepted&~observed,'inferred_internal_fill_mask':inferred_fill&accepted,
        'foreground_or_glass_mask':accepted&foreground,'ray_domain':base['ray_domain'],
        'pre_occluder_filter_opening_hypothesis_mask':pre_filter_accepted,
        'excluded_predicted_solid_occluder_mask':excluded_occluders,
        'occluder_diagnostic_mask':explicit_occluder&gaps,
        'normal_angle':base['normal_angle'],'signed_away':base['signed_away'],'components':records,
        'metadata':{'method':'current_layer_directional_enclosure_plus_bounded_context_recession_completion',
            'config':cfg,'surface_role':surface_role,'opening_annotation_used':False,
            'opening_model_used':False,'photo_specific_parameters_used':False,
            'plane_quality_passed':quality,'accepted_pixels':int(accepted.sum()),
            'uncertain_pixels':int(uncertain.sum()),'rejected_pixels':int(rejected.sum()),
            'observed_recession_pixels':int(observed.sum()),'geometry_confirmed':False,
            'excluded_predicted_solid_occluder_pixels':int(excluded_occluders.sum()),
            'opening_context_ids':resolved['opening'],'facade_column_ids':resolved['facade']+resolved['column'],
            'interpretation':'Predicted footprint candidates. Enclosed support and recess/context are combined; large unsupported outer gaps remain uncertain. Glass and lower-floor objects may remain inside candidate footprints.',
            'limitation':'Single-plane and predicted semantics cannot reliably distinguish open voids, recesses, glass, or all exterior spaces. Directional enclosure may include foreground equipment; no opening accuracy or physical measurement is claimed.'}}
