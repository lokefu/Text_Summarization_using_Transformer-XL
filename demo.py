# Imports
import streamlit as st
import pandas as pd
import base64
from tokenization_transfo_xl import *
from modeling_transfo_xl import *
from modeling_transfo_xl_utilities import *
from modeling_tf_transfo_xl import *
from modeling_tf_transfo_xl_utilities import *

# Load the saved model checkpoint and tokenizer from the `best_checkpoint.pt` file and `best_checkpoint` directory, respectively
model_state_dict = torch.load('best_checkpoint.pt')
tokenizer = TransfoXLTokenizer.from_pretrained('best_checkpoint')

# Add a padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define a function for summarizing text
def summarize(text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(text, truncation='only_first', padding='max_length', max_length=128, return_tensors='pt')
    
    # Generate a summary
    model = TransfoXLLMHeadModel.from_pretrained(pretrained_model_name_or_path='transfo-xl-wt103', state_dict=model_state_dict)
    outputs = model.generate(inputs['input_ids'], pad_token_id=tokenizer.eos_token_id, num_beams=5, max_length=129, early_stopping=True,  no_repeat_ngram_size=2, top_k=2, temperature=1.0)

    # Decode the summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary
 
# Initialize the SessionState object to store summaries
if 'summary_data' not in st.session_state:
    st.session_state['summary_data'] = []

# Define the Streamlit app
def app():
    # Set the webpage title and description
    st.set_page_config(page_title='Text Summarization Demo', page_icon=':memo:', layout='wide')
    st.title('Text Summarization')
    st.write('This page uses our Transformer-XL model checkpoint to generate summaries of text.')
    
    # Create a text input box for user input
    input_text = st.text_area('Enter text to summarize:', height=200)
    
    # Create a button to trigger the text summarization process
    if st.button('Summarize'):

         # Call the summarize function and generate a summary
        summary = summarize(input_text)
        
        # Display the generated summary
        st.subheader('Summary')
        st.write(summary)
 
        # Save the summary and append it to the list in the SessionState object
        st.session_state['summary_data'].append(summary)
        
    # Display previously generated summaries and the recent summary
    if len(st.session_state['summary_data']) > 0:
        with st.expander('Text Summarization Histories'):
            summary_list = [{'Your Previous Summaries and the Recent Summary': summary} for summary in st.session_state['summary_data']]
            df = pd.DataFrame(summary_list)
            df = df.reset_index(drop=True)
            df.index = df.index + 1
            st.dataframe(df, height=210, width=800)
            
            # Create a link for downloading the summaries in a txt file
            txt = "\n".join(st.session_state['summary_data'])
            b64 = base64.b64encode(txt.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="Summary.txt">Download text file</a>'
            st.markdown(href, unsafe_allow_html=True)     

if __name__ == '__main__':
    app()