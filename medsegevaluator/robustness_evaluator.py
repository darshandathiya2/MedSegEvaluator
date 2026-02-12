import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from med_seg_metrics import dice_score
from medical_image_loader import MedicalImageLoader


class RobustnessEvaluator:
    """
    Evaluate segmentation model robustness under image perturbations
    using Dice score and distribution shift analysis.
    """

    def __init__(
        self,
        model,
        loader=None,
        threshold=0.5
    ):
        """
        Parameters
        ----------
        model : tensorflow.keras.Model
            Trained segmentation model.

        loader : MedicalImageLoader, optional
            Image loader instance.

        threshold : float
            Binarization threshold for predictions.
        """

        self.model = model
        self.threshold = threshold

        if loader is None:
            self.loader = MedicalImageLoader(normalize=True)
        else:
            self.loader = loader

        self.perturbations = [
            "gaussian_noise",
            "gaussian_blur",
            "brightness",
            "salt_pepper",
            "rotation_90",
            "rotation_180",
            "rotation_270",
            "horizontal_flip",
            "vertical_flip"
        ]


    # ============================================
    # Perturbations
    # ============================================

    def _horizontal_flip(self, img):
        return np.fliplr(img)

    def _vertical_flip(self, img):
        return np.flipud(img)

    def _gaussian_noise(self, img, mean=0, var=0.001):

        img = img.astype(np.float32)

        if img.max() > 1:
            img /= 255.0

        noise = np.random.normal(mean, np.sqrt(var), img.shape)

        return np.clip(img + noise, 0, 1)


    def _salt_pepper(self, img, amount=0.02):

        img = img.copy()

        num = int(amount * img.size)

        coords = [
            np.random.randint(0, i, num)
            for i in img.shape
        ]

        img[tuple(coords)] = 1

        coords = [
            np.random.randint(0, i, num)
            for i in img.shape
        ]

        img[tuple(coords)] = 0

        return img


    def _brightness(self, img, factor=1.2):

        img = img.astype(np.float32)

        if img.max() > 1:
            img /= 255.0

        return np.clip(img * factor, 0, 1)


    def _gaussian_blur(self, img, ksize=(5, 5)):

        img = img.astype(np.float32)

        if img.max() > 1:
            img /= 255.0

        return cv2.GaussianBlur(img, ksize, 0)


    def _rotate(self, img, angle):
        return np.rot90(img, k=angle // 90)


    def _apply_perturbation(self, img, ptype, is_mask=False):

        if ptype == "horizontal_flip":
            return self._horizontal_flip(img)

        if ptype == "vertical_flip":
            return self._vertical_flip(img)

        if ptype == "gaussian_noise" and not is_mask:
            return self._gaussian_noise(img)

        if ptype == "salt_pepper" and not is_mask:
            return self._salt_pepper(img)

        if ptype == "gaussian_blur" and not is_mask:
            return self._gaussian_blur(img)

        if ptype == "brightness" and not is_mask:
            return self._brightness(img)

        if ptype == "rotation_90":
            return self._rotate(img, 90)

        if ptype == "rotation_180":
            return self._rotate(img, 180)

        if ptype == "rotation_270":
            return self._rotate(img, 270)

        return img


    # ============================================
    # Evaluation
    # ============================================

    def evaluate(
        self,
        image_paths,
        mask_paths,
        output_csv=None
    ):
        """
        Run robustness evaluation.

        Parameters
        ----------
        image_paths : list
            List of image file paths.

        mask_paths : list
            List of ground-truth mask paths.

        output_csv : str, optional
            Path to save results.

        Returns
        -------
        pandas.DataFrame
            Robustness evaluation results.
        """

        assert len(image_paths) == len(mask_paths)

        results = []


        # ----------------------------------
        # Loop over dataset
        # ----------------------------------
        for img_p, gt_p in zip(image_paths, mask_paths):

            img = self.loader.load_image(str(img_p))
            gt = self.loader.load_image(str(gt_p))

            record = {
                "Image_ID": Path(img_p).stem
            }


            # ---------- Original ----------

            x = np.expand_dims(img, axis=0)

            pred = self.model.predict(x, verbose=0)[0, ..., 0]
            pred = (pred > self.threshold).astype("float32")

            dice0 = dice_score(gt, pred)

            record["Dice_original"] = float(dice0)

            pert_values = []


            # ---------- Perturbations ----------

            for p in self.perturbations:

                img_t = self._apply_perturbation(img, p, False)
                gt_t = self._apply_perturbation(gt, p, True)

                x_t = np.expand_dims(img_t, axis=0)

                pred_t = self.model.predict(x_t, verbose=0)[0, ..., 0]
                pred_t = (pred_t > self.threshold).astype("float32")

                dice_t = dice_score(gt_t, pred_t)

                record[f"Dice_{p}"] = float(dice_t)

                pert_values.append(dice_t)


            # ---------- Summary ----------

            mean_p = np.mean(pert_values)
            delta = dice0 - mean_p

            record["Mean_Perturb_Dice"] = float(mean_p)
            record["Delta_Dice"] = float(delta)

            results.append(record)


        # ----------------------------------
        # DataFrame
        # ----------------------------------

        df = pd.DataFrame(results)

        if output_csv:
            df.to_csv(output_csv, index=False)

        return df
