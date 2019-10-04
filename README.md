# An NLP Project on "The Silmarillion" by J.R.R. Tolkien
<img src="Images/Valinor.jpg" width="900" height="400">

# 1 Introduction
## 1.1 Natural Langauge Processing 
Natural Language Processing, or NLP, is within the field of linguistics and refers to a set of techniques for manipulation of natural language, such as speech and text, using software. NLP is a sub-branch of data science and has many applications, though it has been particularly useful in the healthcare industry, with the increased use of Electronic Health Records (EHR). For example, we can now predict risk for suicide and suicidal ideation by applying NLP to EHR's ([1](https://www.nature.com/articles/s41598-018-25773-2)). Companies can apply NLP to social media or product reviews, in order to understand their customers better and ultimately offer a better product or service ([2](https://www.researchgate.net/publication/309691845_A_Review_of_Natural_Language_Processing_Techniques_for_Opinion_Mining_Systems)). Since I don't have ready access to EHR's, I've decided to do this NLP project on one of my favorite books, "The Silmarillion", by J.R.R. Tolkien. 

## 1.2 The Silmarillion
When people hear of J.R.R. Tolkien, his book "The Silmarillion" is typically not the first that comes to mind. After all, it's difficult to compete with books like "The Hobbit" and "The Lord of the Rings", as both are far easier to read and have big budget Hollywood films tied to them. Still, "The Silmarillion" holds a special place in my heart. It lays the foundation for Middle-earth and sets the stage for the more popular books in Tolkien's catalogue. The book is incredibly dense, content-wise, and written in a formal and archaic style, more similar to The Bible than The Lord of the Rings. It's challenging and some find it impenetrable. Though, I can't recommend it more to fans of fantasy and for Tolkien fans, it's simply essential reading. 

Having read both "The Hobbit" and "The Lord of the Rings" at a very early age, I admit to having avoided "The Silmarillion" until about 10 years ago. I was part of a seminar of about 30 other geeks who met online bi-weekly to discuss each chapter of The Silmarillion in detail. The seminar was later released as a podcast ([3](https://tolkienprofessor.com/lectures/courses/silmarillion-seminar/))([4](https://itunes.apple.com/us/course/the-silmarillion-seminar/id599723153)), and for anyone planning to read the book....maybe it will help!

# 2 Data
The full text of "The Silmarillion" used in this project is accessible here ([5](https://archive.org/stream/fegmcfeggerson_gmail_4731/473%20%281%29_djvu.txt)). After scraping the webpage and cleaning the text, I extracted the entire text and the text for each seperate chapter and created a PANDAS dataframe. I then used the modules NLTK and Text Blob to extract some basic information from the text and then the modules Text Blob, VADER, and the NRC Emotion Lexicon to perform sentiment analyses by chapter. All values for sentiment analyses and descriptive text analyses were added to the dataframe, pictured below:

<img src="Images/Data.jpg" width="900" height="190">

# 3 Text Analysis
As a first step, I "tokenized" each word from the book to create a word cloud of the most frequent words across the entire text. Next, I did the same for each chapter and created a table of only the most frequently used nouns, the give a sense of what each chapter is about.


<img src="Images/Silmarillion_wordcloud.png" width="350" height="450">


<img src="Images/Most_freq_nouns.jpg" width="1200" height="1500">
