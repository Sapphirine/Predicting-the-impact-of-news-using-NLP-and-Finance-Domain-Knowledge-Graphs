# Predicting-the-impact-of-news-using-NLP-and-Finance-Domain-Knowledge-Graphs
This is a project written to fulfill the requirements of Advanced Big Data Analytics.

## django_final_proj
This is the folder containing codes up to final project changes.
The html server can be executed by running the manage.py in django folder.

For security reasons:
The secret key for django server is removed in django\big_data\settings.py line 22.
The allowed_hosts is changed in django\big_data\settings.py line 27.
The API key for google knowledge graph is changed to ABC in django\stocks\views.py line 465.
The password for neo4j is changed to root in django\stocks\views.py line 647.

Please generate your unique secret key for django server, update the allowed_hosts and key in your own neo4j server password in order to make it works.

Below are the screenshot of the final web user interface for homepage, stocks analysis and news analysis:
![Screenshot](HomepageScreenshot.png)
![Screenshot](StocksAnalysisScreenshot.png)
![Screenshot](NewsAnalysisScreenshot.png)
