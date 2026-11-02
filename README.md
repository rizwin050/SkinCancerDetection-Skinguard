# Explainable Skin Cancer Detection

This repository contains a small Flask web application that classifies an uploaded skin-lesion image as **Benign** or **Melignant** (the latter is the spelling used by the trained model). It also generates LIME and Grad-CAM visualizations and creates a PDF report from patient details entered in the form.

This is an educational/demo project, not a medical device. Its predictions and visual explanations are not a diagnosis; consult a qualified healthcare professional for medical concerns.

## How it works

1. Open **Verification** and upload a skin image, then enter the patient ID, name, age, date, and gender.
2. The Flask app resizes the image to 224 × 224 and calls the CNN stored in `final_app/skin_xai_model.h5`.
3. The app produces a LIME explanation and a Grad-CAM heatmap, then displays the prediction and generated images.
4. A PDF report is written to `final_app/<patient_name>_<patient_id>/medical_report.pdf` when the app is run from `final_app`.

The app writes intermediate and visualization images to fixed paths (`output.png`, `static/images/output3.png`, `static/images/output4.png`, `static/images/output5.png`, and `static/images/grad.png`). New requests can replace these files.

## Project layout

```text
main_explainable_ai_code.ipynb          # CNN training, evaluation, and LIME notebook
final_app/
  app.py                                # Flask routes and PDF report generation
  model_predict2.py                     # Prediction using skin_xai_model.h5
  explainable_ai_code.py                # LIME explanation generation
  grad_cam_heatmap_analysis_code.py     # Grad-CAM generation
  skin_xai_model.h5                     # Model used by the web app
  model_resnet50.h5                     # Model referenced by the alternate predictor
  templates/                            # Flask/Jinja pages
  static/                               # CSS, JavaScript, and images
  main_explainable_ai_code.ipynb        # Copy of the training/explanation notebook
```

The notebooks expect a `dataset/train/` and `dataset/test/` directory, with class subdirectories, but that dataset is not included here. `grad_cam_heatmap_analysis_code1-Copy1.py` contains additional model/Grad-CAM experiments. The `crop*.html` and `fertilizer*.html` files are template remnants: the current `app.py` does not define crop or fertilizer routes. `model_predict.py` is an alternate predictor and is not used by the current web route.

## Run locally

Python dependency versions are not pinned in this project. Create an environment and install compatible versions of the packages imported by the application and explanation scripts:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install Flask numpy pandas requests Pillow reportlab opencv-python tensorflow keras scikit-image lime matplotlib seaborn scikit-learn
```

Then run the app from its directory so the relative model and asset paths resolve:

```powershell
cd final_app
python app.py
```

Open the local address printed by Flask (typically `http://127.0.0.1:5000/`). The trained `skin_xai_model.h5` file must be present in `final_app`. The included model files use the legacy HDF5 format, and TensorFlow/Keras and Flask versions may need to be compatible with the code.

`app.py` starts Flask with debug mode enabled; this configuration is for local development only and should not be exposed as a production service.
