# Text Summarization using Transformer-XL

This is a text summarization program that uses Transformer-XL, a state-of-the-art language model developed by Google, for fine-tuning on the summarization task. The code is built on the source code of Transformer-XL, with some functions and modules from the Transformers library used for optimization and other purposes.

## Hyperlink to GitHub

```
https://github.com/lokefu/Text_Summarization_using_Transformer-XL
```

## Installation

1. Install the required libraries:

   `````
   pip install torch numpy tqdm transformers streamlit re datasets evaluate rouge_score sacremoses
   `````

2. Download the source code and checkpoints from the GitHub repository: (directly download zip or use git)

   ````
   git clone https://github.com/lokefu/Text_Summarization_using_Transformer-XL.git
   `````

## Dataset

The datasets used in this project are included in the repository and were downloaded from Hugging Face in `dataset.ipynb`. The model was fine-tuned specifically on the pre-processed cnn_dailymail dataset, named with `train_subset.pt` and `val_subset.pt`. As the files exceed the file size limits of GitHub, you need to download our pre-processed dataset from `https://drive.google.com/drive/folders/1E6Yk9Z-q2bkrw5MeAhepzaEHjDlBbQEp?usp=drive_link` before running the code (Remember to put both datasets, two pt. files into the same folder as `model.py` or `model.ipynb`).

## Checkpoints

There are two saved checkpoints after our fine-tuning process, model: a pt. file named with `best_checkpoint.pt`; tokenizer: a folder named with `best_checkpoint`. As the files exceed the file size limits of GitHub, you need to download our checkpoints from `https://drive.google.com/drive/folders/1cvbhKFDuTVUtq-bdQ3A_fvhCHTgkpqd8?usp=sharing` before running the code (Remember to put both checkpoints, one folder and one pt. file into the same folder as `model.py` or `model.ipynb`).

## Usage

To use the program, run the `demo.py` script: (make sure that you have downloaded all the files and folders)

```
streamlit run app.py
```

This will start the Streamlit app and open it in your default web browser. The program will prompt you to enter the input text. You can then enter some text to summarize and click the "Summarize" button to generate a summary. The generated summary will be displayed in a stylized summary box. This demo is supported by our checkpoint through fine-tuning.

To get your own checkpoint, re-train the `model.py` script: (make sure that you have downloaded all the files and folders and you have downloaded the datasets from the link)

```
python model.py
```

This will save your own checkpoint (`best_checkpoint.pt`: model; `best_checkpoint`: tokenizer). There is also a jupyter notebook version of model implementation, `model.ipynb`.

## Acknowledgments

This project builds on the source code of Transformer-XL, which was developed by the Google Brain team. We also use some functions and modules from the Transformers library, developed by Hugging Face, for optimization and other purposes. We would also like to thank Professor Wingyan Chung and TA Yinqiang Zhang for their guidance and support throughout the development of this project.

## License

This project is licensed under the terms of the course license and is intended for educational use only. Redistribution and commercial use of this project are strictly prohibited.

## Contribution

1. Fu Yicheng, Loke (Group Coordination):
   My role was coordinating the project, ensuring that each team member was on track to complete their assigned tasks, providing support as needed, and overseeing the project's progress. In addition, I provided detailed and well-formatted code templates in Python files for each part of the project, making it easier for the team to work with and debug the code.
   Regarding the data part, I provided valuable input on data cleaning and preprocessing, which helped ensure that the dataset was properly prepared for training the Transformer-XL model. For the model part, I gave useful feedback on the Transformer-XL model architecture and fine-tuning script, which helped refine the model and improve its accuracy. As for the demo part, I contributed by providing valuable input on the design and functionality of the demo, ensuring that it met the requirements and was user-friendly.
   Overall, my contributions helped the team work more efficiently and effectively. I also contributed to the coding parts beyond providing Py files, including debugging, explaining the code, and other related coding tasks that helped ensure the project's success. I participated in almost all of the code work for the model part and most of the code work for the data and demo parts. Lastly, I created the official GitHub website, including the README.md file for our project.

2. (Dataset):

3. (Model):

4. (Demo):



## Contact

If you have any questions or comments about this project, please contact us at lokefu@hku.hk.
