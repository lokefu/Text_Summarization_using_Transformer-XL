# Text Summarization using Transformer-XL

This is a text summarization program that uses Transformer-XL, a state-of-the-art language model developed by Google, for fine-tuning on the summarization task. The code is built on the source code of Transformer-XL, with some functions and modules from the Transformers library used for optimization and other purposes.

## Installation

1. Install the required libraries:

   `````
   pip install torch numpy tqdm transformers streamlit re datasets evaluate rouge_score sacremoses

2. Download the source code and checkpoints from the GitHub repository: (directly download zip or use git)

   ````
   git clone https://github.com/lokefu/Text_Summarization_using_Transformer-XL.git

## Usage

To use the program, run the `demo.py` script:

```
streamlit run app.py

```

This will start the Streamlit app and open it in your default web browser. The program will prompt you to enter the input text. You can then enter some text to summarize and click the "Summarize" button to generate a summary. The generated summary will be displayed in a stylized summary box. This demo is supported by our checkpoint through fine-tuning.

To get your own checkpoint, re-train the `model.py` script:

```
python model.py

```

This will save your own checkpoint.

## Dataset

The datasets used in this project are included in the repository and were downloaded from Hugging Face in `dataset.ipynb`. The model was fine-tuned specifically on the cnn_dailymail dataset.

## Acknowledgments

This project builds on the source code of Transformer-XL, which was developed by the Google Brain team. We also use some functions and modules from the Transformers library, developed by Hugging Face, for optimization and other purposes. We would also like to thank Professor Wingyan Chung and TA Yinqiang Zhang for their guidance and support throughout the development of this project.

## License

This project is licensed under the terms of the course license and is intended for educational use only. Redistribution and commercial use of this project are strictly prohibited.

## Contact

If you have any questions or comments about this project, please contact us at lokefu@hku.hk.
