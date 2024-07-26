from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAMParameters():
    """
    points_per_side (int or None): The number of points to be sampled
        along one side of the image. The total number of points is
        points_per_side**2. If None, 'point_grids' must provide explicit
        point sampling.
    points_per_batch (int): Sets the number of points run simultaneously
        by the model. Higher numbers may be faster but use more GPU memory.
    pred_iou_thresh (float): A filtering threshold in [0,1], using the
        model's predicted mask quality.
    stability_score_thresh (float): A filtering threshold in [0,1], using
        the stability of the mask under changes to the cutoff used to binarize
        the model's mask predictions.
    stability_score_offset (float): The amount to shift the cutoff when
        calculated the stability score.
    box_nms_thresh (float): The box IoU cutoff used by non-maximal
        suppression to filter duplicate masks.
    crop_n_layers (int): If >0, mask prediction will be run again on
        crops of the image. Sets the number of layers to run, where each
        layer has 2**i_layer number of image crops.
    crop_nms_thresh (float): The box IoU cutoff used by non-maximal
        suppression to filter duplicate masks between different crops.
    crop_overlap_ratio (float): Sets the degree to which crops overlap.
        In the first crop layer, crops will overlap by this fraction of
        the image length. Later layers with more crops scale down this overlap.
    crop_n_points_downscale_factor (int): The number of points-per-side
        sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
    point_grids (list(np.ndarray) or None): A list over explicit grids
        of points used for sampling, normalized to [0,1]. The nth grid in the
        list is used in the nth crop layer. Exclusive with points_per_side.
    min_mask_region_area (int): If >0, postprocessing will be applied
        to remove disconnected regions and holes in masks with area smaller
        than min_mask_region_area. Requires opencv.
    output_mode (str): The form masks are returned in. Can be 'binary_mask',
        'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
        For large resolutions, 'binary_mask' may consume large amounts of
        memory.
    """
    def __init__(
            self,
            points_per_side = 32,
            points_per_batch = 64,
            pred_iou_thresh = 0.88,
            stability_score_thresh = 0.95,
            stability_score_offset = 1.0,
            crop_n_layers = 0,
            box_nms_thresh = 0.7,
            crop_nms_thresh = 0.7,
            crop_overlap_ratio = 512 / 1500,
            crop_n_points_downscale_factor = 1,
            point_grids = None,
            min_mask_region_area = 0,
            output_mode = "binary_mask"
        ):
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.point_grids = point_grids
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

class InstanceSegmentation():

    def __init__(self, model_type, checkpoint_path, use_cuda, params):
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        if use_cuda:
            self.sam.to(device="cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side = params.points_per_side,
            points_per_batch = params.points_per_batch,
            pred_iou_thresh = params.pred_iou_thresh,
            stability_score_thresh = params.stability_score_thresh,
            stability_score_offset = params.stability_score_offset,
            box_nms_thresh = params.box_nms_thresh,
            crop_n_layers = params.crop_n_layers,
            crop_nms_thresh = params.crop_nms_thresh,
            crop_overlap_ratio = params.crop_overlap_ratio,
            crop_n_points_downscale_factor = params.crop_n_points_downscale_factor,
            point_grids = params.point_grids,
            min_mask_region_area = params.min_mask_region_area,
            output_mode = params.output_mode,
        )
        self.predictor = SamPredictor(self.sam)
        print("InstanceSegmentation SAM model initialized!")

    def segment_all(self, img):
        return self.mask_generator.generate(img)

    def segment_by_roi(self, img, roi_box):
        self.predictor.set_image(img)
        masks, _, _ = self.predictor.predict(
            box = roi_box[None, :],   # 需要增加一个维度来匹配输入形状
            multimask_output = False  # 如果为 True，将返回多个可能的分割结果
        )
        return masks[0]

if __name__ == "__main__":
    instanceSeg = InstanceSegmentation("vit_h", "../../../data/models/sam_vit_h_4b8939.pth", True, SAMParameters())