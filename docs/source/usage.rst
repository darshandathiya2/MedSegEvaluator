Usage
======

**Example**

- MedSefEvaluator example code and data is available in the `GitHub Repository <https://github.com/darshandathiya2/MedSegEvaluator.git>`_.
- The sample data is available in ``MedSegEvaluator/data``
- Use jupyter notebook or google colab to run the 2D or 3D example, located in ``MedSegEvaluator/Examples``

**Performance Analysis**

To visualize the image along with its ground truth and predicted output:

.. code-block:: python

    from perforamce_visualization import visualize_image_contour_3d

    visualize_image_contour_3d(image, gt_image, inf_image, slice_index=57)

To evaluate all performance metrics for the single ground truth and predicted output:

.. code-block:: python

    from performance_metrics import evaluate_all_metrics

    results = evaluate_all_metrics(GT_img, inf_img, voxel_spacing=None)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

To plot the Bland-Altman Plot:

.. code-block:: python

    # Bland-Altman plot for GT volume and predicted volume
    from bland_altman_plot import bland_altman_plot
    
    gt_volume = np.array(df_results['GT_Volume'])
    inf_volume = np.array(df_results['INF_Volume'])

    bland_altman_plot(gt_volume, inf_volume, title="Bland-Altman Plot: Tumor Volume (mm³)",
        xlabel="Mean Volume", ylabel="Difference (Prediction - Ground Truth)", units='mm³')

To compute the global robustness score under perturbation:

.. code-block:: python

    from medicalimageloader import MedicalImageLoader
    from performance_metrics import global_robustness_score
    # Create loader (normalize=True scales intensities to [0,1])
    loader = MedicalImageLoader(normalize=True)

    gt_image = loader.load_image(GROUND_TRUTH_DIR / 'mask_0001.png')
    inf_image = loader.load_image(INFERENCE_DIR / 'pred_0001.png')

    results = global_robustness_score(gt_image, inf_image, D_ref=50.0)
    print(results['GRS'])


    
