from __future__ import annotations
from collections.abc import Mapping
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from src.ceiling_edge_gaps import extract_ceiling_edge_gaps
from src.opening_detection import detect_openings, DEFAULT_CONFIG as OPENING_DEFAULTS
from src.aperture_boundary import propose_aperture_boundary


DEFAULT_CONFIG={
    'strong_reference_minimum_inlier_ratio':.75,
    'minimum_window_pixels':64,
    'minimum_window_recession_fraction':.60,
    'window_recession_distance_multiplier':2.,
    'minimum_localized_body_image_fraction':.001,
    'maximum_window_candidate_domain_fraction':.20,
    'maximum_boundary_components':4,
    'enable_rgb_boundary':False,
}


def validated_config(config=None):
    if config is not None and not isinstance(config,Mapping):
        raise ValueError('ceiling_v2 config must be a mapping')
    cfg={**DEFAULT_CONFIG,**(config or {})}
    if set(cfg)-set(DEFAULT_CONFIG):
        raise ValueError('Unknown ceiling_v2 configuration')
    for key,value in cfg.items():
        if key=='enable_rgb_boundary':
            if not isinstance(value,(bool,np.bool_)):
                raise ValueError(key+' must be boolean')
            continue
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or value<0:
            raise ValueError(key+' must be finite nonnegative numeric')
        if 'fraction' in key or 'ratio' in key:
            if value>1:
                raise ValueError(key+' must be in [0,1]')
        if key in ('minimum_window_pixels','maximum_boundary_components'):
            if int(value)!=value or value<1:
                raise ValueError(key+' must be a positive integer')
            cfg[key]=int(value)
    return cfg


def detect_ceiling_slab(semantic,points,normals,valid,K,reference_plane,*,
                          classes=None,labels=None,supplemental_seeds=None,rgb=None,config=None):
    cfg=validated_config(config)
    semantic=np.asarray(semantic)
    if semantic.ndim!=2 or semantic.dtype!=bool:
        raise ValueError('semantic must be boolean H x W ceiling seed')
    shape=semantic.shape
    empty=np.zeros(shape,bool)
    if (classes is None)!=(labels is None):
        raise ValueError('classes and labels must be supplied together')
    if classes is not None:
        classes=np.asarray(classes)
        if classes.shape!=shape or not np.issubdtype(classes.dtype,np.integer) or not isinstance(labels,Mapping):
            raise ValueError('classes must be integer H x W with id-to-label mapping')
    if rgb is None:
        rgb=np.zeros((*shape,3),np.uint8)
        boundary_enabled=False
    else:
        rgb=np.asarray(rgb)
        if rgb.shape!=(*shape,3) or rgb.dtype!=np.uint8:
            raise ValueError('rgb must be uint8 H x W x 3 on the same frame')
        boundary_enabled=bool(cfg['enable_rgb_boundary'])
    result={'opening_hypothesis_mask':empty.copy(),'bounded_uncertain_mask':empty.copy(),
        'ambiguous_opening_mask':empty.copy(),'observed_recess_mask':empty.copy(),
        'inferred_void_mask':empty.copy(),'recovered_slab_support':semantic.copy(),
        'raw_gap_mask':empty.copy(),'semantic_window_seed_mask':empty.copy(),
        'localized_geometry_body_mask':empty.copy(),'rgb_boundary_proposal_mask':empty.copy(),
        'components':[],'metadata':{'method':'strong_original_reference_ceiling_edge_gaps_or_independent_window_recession',
            'config':cfg,'surface_role':'ceiling','physical_through_void_confirmed':False,
            'opening_annotation_read':False,'manual_opening_mask_used':False,
            'multi_offset_support_union_used':False,'local_plane_upgrades_global_quality':False,
            'accepted_pixels':0,'bounded_uncertain_pixels':0,'status':'unknown_missing_original_reference'}}
    if not semantic.any() or reference_plane is None:
        return result
    coefficients=reference_plane.get('plane',reference_plane)
    plane={**coefficients,'metadata':reference_plane.get('metadata',coefficients.get('metadata',{}))}
    quality=plane['metadata'].get('quality_passed') is True
    inlier=float(reference_plane.get('inlier_ratio',plane['metadata'].get('inlier_ratio_original_role',0.)))
    strong=quality and np.isfinite(inlier) and inlier>=cfg['strong_reference_minimum_inlier_ratio']
    role=reference_plane.get('surface_role',plane['metadata'].get('surface_role'))
    if role is not None and role!='ceiling':
        raise ValueError('Original reference must be for ceiling, not floor')
    edge=extract_ceiling_edge_gaps(semantic,points,normals,valid,K,plane,supplemental_seeds=supplemental_seeds)
    result['recovered_slab_support']=edge['recovered_support']
    result['raw_gap_mask']=edge['raw_gap_mask']
    usable=np.asarray(valid)&np.isfinite(points).all(axis=-1)&(np.asarray(points)[...,2]>0)
    threshold=float(plane['metadata']['effective_plane_distance_threshold'])
    away=edge['signed_away']
    recession=usable&(away>threshold*cfg['window_recession_distance_multiplier'])
    if strong:
        result['opening_hypothesis_mask']|=edge['opening_candidate_mask']
        result['components'] += [{**record,'route':'reference_edge',
                                  'source_detector_status':record['status'],'status':'unconfirmed_footprint_candidate'}
                                for record in edge['components'] if record['accepted_as_candidate']]
    lengths=np.linalg.norm(normals,axis=-1)
    normal_usable=usable&np.isfinite(normals).all(axis=-1)&(lengths>1e-8)
    angle=np.full(shape,np.nan)
    normal=np.asarray(plane['normal'])
    angle[normal_usable]=np.degrees(np.arccos(np.clip(np.abs(np.asarray(normals)[normal_usable]@normal/lengths[normal_usable]),0,1)))
    local_cfg={**OPENING_DEFAULTS,'support_mode':'orientation_local_depth'}
    local=detect_openings(rgb,semantic,points,valid,angle,plane,local_cfg)
    localized=local['final_opening_mask'].copy()
    nc,components,stats,_=cv2.connectedComponentsWithStats(localized.astype('uint8'),8)
    minimum=max(cfg['minimum_window_pixels'],int(np.ceil(semantic.size*cfg['minimum_localized_body_image_fraction'])))
    for index in range(1,nc):
        if stats[index,cv2.CC_STAT_AREA]<minimum:
            localized[components==index]=False
    result['localized_geometry_body_mask']=localized
    if not strong:
        result['bounded_uncertain_mask']|=localized
        result['components'] += [{**record,'route':'orientation_local_depth',
                                  'source_detector_status':record['status'],'status':'weak_global_reference_localized_unknown'}
                                for record in local['components'] if record['status']=='probable_opening']
    window_ids=[]
    if labels is not None:
        for class_id,label in labels.items():
            if not isinstance(label,str):
                raise ValueError('Semantic label names must be strings')
            if label.strip().casefold() in ('windowpane','sky'):
                window_ids.append(int(class_id))
    windows=(np.isin(classes,window_ids) if classes is not None else empty.copy())
    windows&=edge['search_domain']&usable
    candidate_seed=windows&recession
    count,components,stats,_=cv2.connectedComponentsWithStats(windows.astype('uint8'),8)
    boundary_count=0
    boundary_growth_used=False
    for index in range(1,count):
        window=components==index
        area=int(window.sum())
        recess_count=int(np.count_nonzero(window&candidate_seed))
        if area<cfg['minimum_window_pixels'] or recess_count/area<cfg['minimum_window_recession_fraction']:
            continue
        if area/max(1,int(edge['search_domain'].sum()))>cfg['maximum_window_candidate_domain_fraction']:
            continue
        seed=binary_fill_holes(window)&edge['search_domain']
        result['semantic_window_seed_mask']|=seed
        boundary=None
        if boundary_enabled and boundary_count<cfg['maximum_boundary_components']:
            boundary=propose_aperture_boundary(rgb,seed,ceiling=semantic,envelope=edge['search_envelope'],support=edge['recovered_support'])
            boundary_count+=1
            proposed=boundary['proposal_mask']&edge['search_domain']
            proposal_recession_fraction=np.count_nonzero(proposed&recession)/max(1,int(proposed.sum()))
            if proposed.any() and proposal_recession_fraction>=cfg['minimum_window_recession_fraction']:
                result['rgb_boundary_proposal_mask']|=proposed
                if quality and boundary['status']=='supported_boundary':
                    seed=proposed
                    boundary_growth_used=True
                else:
                    result['bounded_uncertain_mask']|=proposed
        if quality:
            result['opening_hypothesis_mask']|=seed
            status='semantic_window_recession_partial_footprint_candidate'
        else:
            result['bounded_uncertain_mask']|=seed
            status='semantic_window_recession_weak_reference_unknown'
        result['components'].append({'route':'semantic_window_positive_recession','status':status,
            'area_pixels':int(seed.sum()),'semantic_window_pixels':area,'positive_recession_pixels':recess_count,
            'positive_recession_fraction':recess_count/area,'bbox_xywh':stats[index,:4].tolist(),
            'RGB_boundary_status':None if boundary is None else boundary['status'],
            'RGB_boundary_evidence':None if boundary is None else boundary['metadata'],
            'physically_confirmed':False})
    candidate=result['opening_hypothesis_mask']
    result['bounded_uncertain_mask']&=~candidate
    result['ambiguous_opening_mask']=result['bounded_uncertain_mask'].copy()
    result['observed_recess_mask']=candidate&recession
    result['inferred_void_mask']=candidate&~result['observed_recess_mask']
    result['metadata'].update({'status':'unconfirmed_candidate_found' if candidate.any() else
        ('localized_unknown_requires_review' if result['bounded_uncertain_mask'].any() else 'no_reliable_candidate_found'),
        'original_reference_quality_passed':quality,'original_reference_inlier_ratio':inlier,
        'strong_original_reference':strong,'edge_candidate_pixels':int(edge['opening_candidate_mask'].sum()),
        'accepted_pixels':int(candidate.sum()),'bounded_uncertain_pixels':int(result['bounded_uncertain_mask'].sum()),
        'semantic_window_qualified_pixels':int(result['semantic_window_seed_mask'].sum()),
        'RGB_boundary_attempts':boundary_count,'RGB_boundary_used_for_growth':boundary_growth_used,
        'RGB_boundary_proposal_generated':bool(result['rgb_boundary_proposal_mask'].any()),
        'edge_metadata':edge['metadata'],
        'limitation':'All masks are candidates or localized unknown footprints. Glass, multiple ceiling levels and coffers can remain indistinguishable; a visible window seed can cover only the inner glass, not the full mouth. No accuracy or physical void measurement is claimed.'})
    return result
