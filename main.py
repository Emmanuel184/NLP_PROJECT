from Theme_Classification import ThemeClassifier
import streamlit as st
import matplotlib.pyplot as plt
from character_chatbot import CharacterChatbot
import os

def get_themes(theme_list, subtitles_path):
    
    theme_list = theme_list.split(",")
    theme_classifier = ThemeClassifier(theme_list=theme_list)
    output_df = theme_classifier.get_themes(subtitles_path)

    theme_list = [theme for theme in theme_list if theme != "dialogue"]
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ["Theme", "Score"]

    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Theme",
        tooltip=["Theme", "Score"],
        vertical=False,
        width=500,
        height=200
    )

    return output_df

def chat_with_bot(message, history):
    charcater_chatbot = CharacterChatbot("avacado10/office_llama_3_8b", huggingface_token=os.getenv("HUGGINGFACE_TOKEN"))
    output = charcater_chatbot(message=message, history=history)
    output = output["content"].strip()
    return output
def main():

    chat_with_bot("something", ["nothing"])

    col_left, col_right = st.columns(2)
    plot_placeholder = col_left.empty()
    data = get_themes("hope, betrayal, lost, love", "/Users/homedirel/ML/DATA/NLP_DATA/4_season_office")
    fig, ax = plt.subplots()
    ax.barh(data['Theme'], data['Score'], color='skyblue')
    ax.set_xlabel('Score')
    ax.set_ylabel('Theme')
    plot_placeholder.pyplot(fig)

    with col_right:
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
            """, 
            unsafe_allow_html=True
        )
        theme_input = st.text_input("Input themes to compute", value="hate, anger")
        generate_button = st.button("Generate Bar Plot")
        st.markdown("</div>", unsafe_allow_html=True)


    if generate_button:
        new_data = get_themes(theme_input, "/Users/homedirel/ML/DATA/NLP_DATA/4_season_office")
        fig, ax = plt.subplots()
        ax.barh(new_data['Theme'], new_data['Score'], color='skyblue')
        ax.set_xlabel('Score')
        ax.set_ylabel('Theme')
        plot_placeholder.pyplot(fig)
        



if __name__ == "__main__":
    main()