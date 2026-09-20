# Conicoid relative error (later validation)

This folder is the future home of a Kerr-vs-conicoid accuracy report on the
Blender synthetic eye grid. **Not run in the current slice.**

Reuse the existing simulation table:

[`development/kerr_relative_error/ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv`](../kerr_relative_error/ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv)

Columns: `center_x`, `center_y`, `axs_1`, `axs_2`, `ellipse_phi`, `x_angle`, `y_angle`.
Geometry: **8 mm** eye diameter, **13 mm** camera distance.

Compare angular error of:

1. Naïve Safaee-Rad unprojection (per-frame, no sphere)
2. Swirski initialisation (`fit_eye_and_gaze`, this package)
3. Kerr (`BlockSync.kerr`) at image-centre ref `(320, 240)`, as in Figure S1a

That comparison is how we will judge whether the conicoid method beats Kerr’s
residuals. Image-based Swirski refine and Dierkes 2019 refraction correction
stay out of scope until this baseline exists.
