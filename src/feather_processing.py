import gc
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import SAM


@dataclass
class ProcessResult:
    image_path: str
    success: bool
    reason: str = ""


class FeatherProcessor:
    def __init__(self) -> None:
        load_dotenv()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(self.device)
        self.sam_model = SAM("sam2.1_b.pt")

        self.has_vlm = False
        self.vlm_model = None
        self.vlm_processor = None
        try:
            from mlx_vlm import load as mlx_load

            self.vlm_model, self.vlm_processor = mlx_load("mlx-community/Qwen2-VL-7B-Instruct-4bit")
            self.has_vlm = True
        except Exception as exc:  # noqa: BLE001
            logging.warning("MLX VLM unavailable; continuing without QA model: %s", exc)

    def _infer_metadata(self, image_path: str) -> tuple[str, str]:
        bird_id = "UNKNOWN"
        date = "UNKNOWN"
        if "A1383" in image_path:
            bird_id = "A1383"
        if "1999" in image_path:
            date = "1999-05-10"
        if "2000" in image_path:
            date = "2000-06-12"
        return bird_id, date

    def _cleanup(self) -> None:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    def process_image(self, image_path: str, output_dir: str) -> ProcessResult:
        bird_id, date = self._infer_metadata(image_path)

        try:
            img_pil = Image.open(image_path).convert("RGB")
            inputs = self.dino_processor(images=img_pil, text="bird feather.", return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.dino_model(**inputs)

            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[img_pil.size[::-1]],
            )[0]

            boxes_with_scores: list[list[float]] = []
            img_area = img_pil.size[0] * img_pil.size[1]
            for score, box in zip(results["scores"], results["boxes"]):
                if float(score) <= 0.25:
                    continue
                b = box.tolist()
                width = b[2] - b[0]
                height = b[3] - b[1]
                if (width * height) > (img_area * 0.45):
                    continue
                shrink = 0.02
                b[0] += width * shrink
                b[1] += height * shrink
                b[2] -= width * shrink
                b[3] -= height * shrink
                boxes_with_scores.append([b[0], b[1], b[2], b[3], float(score)])

            if not boxes_with_scores:
                return ProcessResult(image_path=image_path, success=False, reason="no_boxes")

            boxes_tensor = torch.tensor([b[:4] for b in boxes_with_scores], dtype=torch.float32)
            scores_tensor = torch.tensor([b[4] for b in boxes_with_scores], dtype=torch.float32)
            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, 0.3)
            boxes = [boxes_with_scores[int(i)][:4] for i in keep_indices]

            if not boxes:
                return ProcessResult(image_path=image_path, success=False, reason="no_boxes_after_nms")

            sam_results = self.sam_model(image_path, bboxes=boxes, device=self.device, verbose=False)
            sam_res = sam_results[0]
            if sam_res.masks is None:
                return ProcessResult(image_path=image_path, success=False, reason="no_masks")

            img = cv2.imread(image_path)
            feathers: list[dict] = []

            for mask in sam_res.masks.data.cpu().numpy():
                mask_resized = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                y_max = y + h - int(h * 0.12)

                final_mask = np.zeros_like(img)
                cv2.drawContours(final_mask, [contour], -1, (255, 255, 255), -1)
                final_mask[y_max:, :] = 0

                clean_feather = cv2.bitwise_and(img, final_mask)
                feather_crop = clean_feather[y:y_max, x : x + w]
                feathers.append({"x": x, "crop": feather_crop})

            if not feathers:
                return ProcessResult(image_path=image_path, success=False, reason="no_feathers_after_sam")

            os.makedirs(output_dir, exist_ok=True)
            feathers.sort(key=lambda item: item["x"])
            for idx, feather in enumerate(feathers[:5], start=1):
                crop = feather["crop"]
                img_yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                crop_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                filename = f"{bird_id}_{date}_Feather_{idx}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), crop_eq)

            return ProcessResult(image_path=image_path, success=True)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to process image %s", image_path)
            return ProcessResult(image_path=image_path, success=False, reason=str(exc))
        finally:
            self._cleanup()
