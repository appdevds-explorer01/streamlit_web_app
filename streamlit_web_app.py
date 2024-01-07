# Core Packages
import streamlit as st
st.set_page_config(page_title="NLP Examples", page_icon="", layout="centered", initial_sidebar_state='auto')

# NLP Packages
from textblob import TextBlob
import spacy
from gensim.summarization import summarize
import neattext as nt

# Visualization Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud

def plot_wordcloud(text_in):
    mywordcloud = WordCloud().generate(text_in)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

def text_analyzer(text_in):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(text_in)
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
    return allData

def detect_language(text_in): 
    from langdetect import detect_langs
    try: 
        langs = detect_langs(text_in) 
        for item in langs: 
            return item.lang
    except: return "err", 0.0 

def main():
    """ NLP Application with Streamlit and TextBlob"""
    title_templ = """
    <div style="background-color:black;padding:6px;">
    <h2 style="color:white">NLP Examples</h2>
    </div>
    """
    st.markdown(title_templ,unsafe_allow_html=True)
    st.sidebar.image("images/gen-ai-img.jpg", use_column_width=True)

    subheader_templ = """
    <div style="background-color:gray;padding:8px;">
    <h3 style="color:black">Natural Language Processing</h3>
    </div>
    """
    st.markdown(subheader_templ,unsafe_allow_html=True)

    activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("Menu",activity)

	# Text Analysis CHOICE
    if choice == 'Text Analysis':
        st.subheader("Text Analysis")
        st.write("")
        st.write("")
        raw_text = st.text_area("Write some text below", "Enter some text in English...",height=250)
        if st.button("Analyze"):
            if len(raw_text) == 0:
                st.warning("Enter a Text...")
            else:
                lang = detect_language(raw_text)
                if lang != 'en':
                    st.warning("Enter a Text in English...")
                else:
                    st.info("Basic Functions...")
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("Basic Information"):
                            st.success("Text Statistics")
                            word_desc = nt.TextFrame(raw_text).word_stats()
                            result_desc = {"Length of Text": word_desc['Length of Text'],
                                           "Number of Vowels": word_desc['Num of Vowels'],
                                           "Number of Consonants": word_desc['Num of Consonants'],
                                           "Number of Stopwords": word_desc['Num of Stopwords']}
                            st.write(result_desc)

                        with st.expander("Stopwords..."):
                            st.success("Stop words list")
                            stop_words = nt.TextExtractor(raw_text).extract_stopwords()
                            st.write(stop_words)

                    with col2:
                        with st.expander("Processed Text"):
                            st.success("Stopwords Excluded Text")
                            processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
                            st.write(processed_text)

                        with st.expander("Wordcloud..."):
                            st.success("Wordcloud")
                            plot_wordcloud(raw_text)

                    st.write("")
                    st.write("")                    
                    st.info("Advanced Features...")
                    col3, col4 = st.columns(2)
                    with col3:
                        with st.expander("Tokens & Lemmas"):
                            st.write("Tokens & Lemmas")
                            text_for_tandl = str(nt.TextFrame(raw_text).remove_stopwords())
                            text_for_tandl = str(nt.TextFrame(text_for_tandl).remove_puncts())
                            text_for_tandl = str(nt.TextFrame(text_for_tandl).remove_special_characters())
                            tandl_text = text_analyzer(text_for_tandl)
                            st.json(tandl_text)

                    with col4:
                        with st.expander("Summarize"):
                            st.success("Summarized Text")
                            summary_text = summarize(raw_text, ratio=0.5)
                            if summary_text != "":
                                st.success(summary_text)
                            else:
                                st.warning("Enter a longer piece of text...")

    # Translation CHOICE
    if choice == 'Translation':
        st.subheader("Text Translation")
        st.write("")
        st.write("")
        raw_text = st.text_area("Write some text below for translation", "Enter some text in English...",height=250)
        if len(raw_text) < 3:
            st.warning("Enter text that is at least 3 characters ...")
        else:
            blob = TextBlob(raw_text)
            lang = detect_language(raw_text)
            translation_options = st.selectbox("Select a language to translate into...", ['Chinese', 'English', 'French', 'Italian', 'Tamil', 'Malayalam', 'Spanish', 'Japanese'])
            if st.button("Translate"):
                if translation_options == 'Italian' and lang != 'it':
                    st.text("Translating to Italian...")
                    translated_text = blob.translate(from_lang=lang, to='it')
                elif translation_options == 'Spanish' and lang != 'es':
                    st.text("Translating to Spanish...")
                    translated_text = blob.translate(from_lang=lang, to='es')
                elif translation_options == 'Chinese' and lang != 'zh-CN':
                    st.text("Translating to Chinese...")
                    translated_text = blob.translate(from_lang=lang, to='zh-CN')
                elif translation_options == 'Tamil' and lang != 'ta':
                    st.text("Translating to Tamil...")
                    translated_text = blob.translate(from_lang=lang, to='ta')
                elif translation_options == 'French' and lang != 'fr':
                    st.text("Translating to French...")
                    translated_text = blob.translate(from_lang=lang, to='fr')
                elif translation_options == 'English' and lang != 'en':
                    st.text("Translating to English...")
                    translated_text = blob.translate(from_lang=lang, to='en')
                elif translation_options == 'Malayalam' and lang != 'ml':
                    st.text("Translating to English...")
                    translated_text = blob.translate(from_lang=lang, to='ml')
                elif translation_options == 'Japanese' and lang != 'ja':
                    st.text("Translating to Japanese...")
                    translated_text = blob.translate(from_lang=lang, to='ja')                    
                else:
                    translated_text = "Text is already in " + "'" + lang + "'"
                st.success(translated_text)
              
    # Sentiment Analysis CHOICE
    if choice == 'Sentiment Analysis':

        st.subheader("Sentiment Analysis")
        st.write("")
        st.write("")

        raw_text = st.text_area("Write text below for sentiment analysis", "Enter some text in English...",height=250)
        if st.button("Analyze Sentiment"):
            if len(raw_text) == 0:
                st.warning("Enter some text for analyzing the sentiment...")
            else:
                blob = TextBlob(raw_text)
                lang = detect_language(raw_text)
                if lang != 'en':
                    translated_text = blob.translate(from_lang=lang, to='en')
                    blob = TextBlob(str(translated_text))
                
                sentiment_result = blob.sentiment
                st.info("Sentiment Polarity: {}".format(sentiment_result.polarity))
                st.info("Sentiment Subjectivity: {}".format(sentiment_result.subjectivity))

    # About CHOICE
    if choice == 'About':
        st.subheader("About")
        st.write("")
        st.write("")

        st.markdown("""
        ### NLP Examples (App with Streamlit and TextBlob)

        ##### By
        + **DS**
        + [abc@gmail.com](mailto:abc@gmail.com)
        """)

if __name__ == '__main__':
    main()
