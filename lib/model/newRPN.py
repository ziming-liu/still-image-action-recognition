import mmcv

from mmdet.models import RPN
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet.models import builder
class newRPN(RPN):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 bbox_roi_extractor=None,
                 pretrained=None):
        super(newRPN, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.rpn_head = builder.build_rpn_head(rpn_head)
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        rois = bbox2roi(proposal_list)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if rescale:
            for proposals, meta in zip(proposal_list, img_meta):
                proposals[:, :4] /= meta['scale_factor']
        # TODO: remove this restriction
        return proposal_list[0].cpu().numpy(),roi_feats
