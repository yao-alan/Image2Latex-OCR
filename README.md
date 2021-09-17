# Image2Latex-OCR
Trained CRNN that outputs LaTeX source text given an input image.

Evaluated on IM2LATEX-100K test images of less than 641x100 px (WxH); averaged a **0.48 BLEU-4** score and edit distance of **56**.

Run the model with
``
  python main.py
    --path <path_to_images>
    --model <path_to_model_parameters>
``.
Alternatively, to view images, run ``main.ipynb``. If optional parameters are excluded, images and model parameters are assumed to be in ``./images`` and ``./saved/run4/epoch1``.
