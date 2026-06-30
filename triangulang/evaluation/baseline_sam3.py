"""Native SAM3 baseline wrapper, shared by evaluation and demos."""
import torch
import torch.nn.functional as F

from sam3.model.data_misc import FindStage


class BaselineSAM3Wrapper(torch.nn.Module):
    """Wrapper around native SAM3 for baseline comparison.

    Runs SAM3's own text-prompted segmentation (encoder + decoder + seghead)
    without GASA or cross-view fusion. Matches TrianguLangModel's forward
    interface so the benchmark/demo loops work unchanged.

    Args:
        sam3_model: native SAM3 model.
        da3_model: optional DA3 model. When provided, forward also runs DA3 and
            returns its depth (used by the demo for 3D unprojection).
        resolution: SAM3 input resolution.
        use_point_prompts: when True, forward accepts point_prompts/point_labels
            and feeds them to SAM3 (used by the demo's click/point mode).
    """

    def __init__(self, sam3_model, da3_model=None, resolution=1008, use_point_prompts=False):
        super().__init__()
        self.sam3 = sam3_model
        self.da3 = da3_model
        self.resolution = resolution
        self.use_point_prompts = use_point_prompts
        self.mask_selection = 'confidence'
        self.use_iou_head = False

    @torch.no_grad()
    def forward(self, images, text_prompts, gt_masks=None,
                gt_intrinsics=None, gt_extrinsics=None,
                point_prompts=None, point_labels=None, **kwargs):
        device = images.device
        B = images.shape[0]

        depth = None
        if self.da3 is not None:
            view_list = [images[i] for i in range(B)]
            depth = self.da3.inference(image=view_list, process_res=518).depth

        sam3_images = (images - 0.5) / 0.5
        if sam3_images.shape[-2:] != (self.resolution, self.resolution):
            sam3_images = F.interpolate(sam3_images, size=(self.resolution, self.resolution),
                                        mode='bilinear', align_corners=False)

        backbone_out = {"img_batch_all_stages": sam3_images}
        backbone_out.update(self.sam3.backbone.forward_image(sam3_images))

        text_out = self.sam3.backbone.forward_text(text_prompts, device=device)
        backbone_out.update(text_out)

        input_points = input_points_mask = None
        if self.use_point_prompts and point_prompts is not None and point_labels is not None:
            input_points = point_prompts * self.resolution  # normalized -> SAM3 pixels
            input_points_mask = point_labels

        find_input = FindStage(
            img_ids=torch.arange(B, device=device, dtype=torch.long),
            text_ids=torch.arange(B, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=input_points, input_points_mask=input_points_mask,
        )
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=B)

        outputs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )

        pred_masks = outputs['pred_masks']
        scores = outputs['pred_logits'].sigmoid().squeeze(-1)
        if 'presence_logit_dec' in outputs:
            scores = scores * outputs['presence_logit_dec'].sigmoid()

        best_idx = scores.argmax(dim=-1)
        batch_idx = torch.arange(B, device=device)
        best_masks = pred_masks[batch_idx, best_idx]

        result = {
            'pred_masks': best_masks.unsqueeze(1),
            'all_masks': pred_masks,
        }
        if self.da3 is not None:
            result['depth'] = depth.unsqueeze(1)
            result['iou_pred'] = None
        return result
