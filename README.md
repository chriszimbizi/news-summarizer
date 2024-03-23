## News Summarizer

This project utilizes OpenAI's GPT-3.5 model to summarize news articles based on user-provided topics. It fetches news data from the News API and generates summaries using the OpenAI Assistant. The summarized news articles are then displayed to the user through a Streamlit interface.

### Setup

Before running the project, make sure to set up the environment variables for `NEWS_API_KEY` and `OPENAI_API_KEY` in a `.env` file located in the `data` directory. These keys are required for accessing the News API and OpenAI services.

### Installation

1. Clone the repository:

```
git clone https://github.com/your_username/your_repository.git
```

2. Navigate to the project directory:

```
cd your_repository
```

3. Install dependencies:

```
pip install -r requirements.txt
```

### Usage

Run the application by executing the `main.py` file:

```
streamlit run main.py
```

The Streamlit interface will open in your default web browser. Enter a topic of interest in the provided text input and click the "Run Assistant" button. The assistant will summarize the latest news on the given topic, and the summaries will be displayed on the interface.

### Functionality

- **Fetching News**: The application fetches news articles from the News API based on the user-provided topic.
- **Summarization**: Utilizing the OpenAI Assistant, the application generates summaries for the fetched news articles.
- **User Interaction**: Users can interact with the application through the Streamlit interface by providing topics and receiving summarized news articles.

### Screenshot

<img alt="Screenshot of Streamlit news summarizer interface" src="https://github.com/chriszimbizi/news-summarizer/assets/121321293/a89a2e68-3483-48ac-a02c-baf7d68b539b" width="400">

### Acknowledgments

- OpenAI for providing the GPT-3.5 model and Assistant services.
- News API for providing access to news articles.
- Streamlit for the user interface framework.
- VinciBits for the tutorial on FreeCodeCamp's YouTube channel, which served as a reference for developing this project.
