Quickstart
==========

The simplest pipeline:

.. code-block:: python

    from perforamce_visualization import visualize_image_contour_3d

    visualize_image_contour_3d(image, gt_image, inf_image, slice_index=57)

or 

.. code-block:: python

    from performance_metrics import evaluate_all_metrics

    results = evaluate_all_metrics(GT_img, inf_img, voxel_spacing=None)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    


