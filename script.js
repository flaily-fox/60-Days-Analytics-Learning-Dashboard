// Data for the 60-Day Data Analytics Roadmap
const roadmapData = [
    // --- REVISED WEEK 1: Python & SQL Fundamentals (7 days) ---
    { day: 1, week: 1, topic: 'Python Fundamentals', sub: 'Environment Setup & Basics', concepts: ['Install Python, Jupyter', 'Variables, Data Types', 'Operators'], time: 3, resource: 'https://www.youtube.com/watch?v=_uQrJ0TkZlc', project: 'Simple calculator script.' },
    { day: 2, week: 1, topic: 'Python Fundamentals', sub: 'Control Flow', concepts: ['If-else statements', 'For & While loops', 'Break & Continue'], time: 3, resource: 'https://www.freecodecamp.org/news/python-if-else-statement-conditional-statements-explained/', project: 'Guess the number game.' },
    { day: 3, week: 1, topic: 'Python Fundamentals', sub: 'Data Structures & Functions', concepts: ['Lists, Tuples, Dictionaries', 'Sets', 'String manipulation', 'Defining functions', 'Modules'], time: 4, resource: 'https://www.youtube.com/watch?v=R-HLU9A5Q-w', project: 'Simple contact book with functions.' },
    { day: 4, week: 1, topic: 'SQL Fundamentals', sub: 'Basic Queries', concepts: ['SELECT, FROM, WHERE', 'ORDER BY, LIMIT', 'Basic operators'], time: 3, resource: 'https://www.youtube.com/watch?v=p3ADB7k3k_g', project: 'Query a sample database for basic data.' },
    { day: 5, week: 1, topic: 'SQL Fundamentals', sub: 'Joins & Aggregations', concepts: ['INNER, LEFT, RIGHT Joins', 'GROUP BY', 'COUNT, SUM, AVG'], time: 4, resource: 'https://www.w3schools.com/sql/sql_join.asp', project: 'Join two tables and calculate summary stats.' },
    { day: 6, week: 1, topic: 'SQL Fundamentals', sub: 'Advanced SQL & Subqueries', concepts: ['Subqueries', 'CASE statements', 'Window Functions (intro)'], time: 4, resource: 'https://www.sqlshack.com/sql-subqueries-in-depth/', project: 'Solve intermediate SQL problems online.' },
    { day: 7, week: 1, topic: 'Week 1 Review', sub: 'Consolidation & Practice', concepts: ['Review Python basics & SQL', 'Setup for Week 2'], time: 2, resource: 'https://www.hackerrank.com/domains/python', project: 'Solve 5 easy Python and 5 easy SQL problems.' },
    
    // --- REVISED WEEK 2: Core Python Libraries (7 days) ---
    { day: 8, week: 2, topic: 'Core Libraries', sub: 'NumPy Basics', concepts: ['Ndarray object', 'Creating arrays', 'Indexing & Slicing'], time: 3, resource: 'https://www.youtube.com/watch?v=Qp5qLg2-hBs', project: 'Create and manipulate 1D and 2D arrays.' },
    { day: 9, week: 2, topic: 'Core Libraries', sub: 'NumPy Operations', concepts: ['Vectorized operations', 'Broadcasting', 'Mathematical functions'], time: 3, resource: 'https://numpy.org/doc/stable/user/basics.broadcasting.html', project: 'Perform mathematical operations on arrays of different shapes.' },
    { day: 10, week: 2, topic: 'Core Libraries', sub: 'Intro to Pandas', concepts: ['Series & DataFrame', 'Reading data (CSV)', 'Inspecting data (`.head`, `.info`)'], time: 4, resource: 'https://www.youtube.com/watch?v=zmdjNSmRXF4', project: 'Load a CSV file into a DataFrame and display info.' },
    { day: 11, week: 2, topic: 'Core Libraries', sub: 'Pandas Data Selection', concepts: ['Selection with `[]`, `.loc`, `.iloc`', 'Conditional filtering'], time: 4, resource: 'https://pandas.pydata.org/docs/user_guide/indexing.html', project: 'Select rows/cols from a dataset based on conditions.' },
    { day: 12, week: 2, topic: 'Core Libraries', sub: 'Pandas Data Cleaning', concepts: ['Handling missing values (`.isnull`, `.fillna`, `.dropna`)', 'Removing duplicates'], time: 3, resource: 'https://www.geeksforGeeks.org/working-with-missing-data-in-pandas/', project: 'Clean a messy dataset (nulls, duplicates).' },
    { day: 13, week: 2, topic: 'Core Libraries', sub: 'Pandas GroupBy & Merging', concepts: ['`groupby()` operation', 'Aggregation functions (`.agg`)', 'Merging DataFrames (`.merge`)'], time: 4, resource: 'https://realpython.com/pandas-groupby/', project: 'Calculate summary stats by category, then merge data.' },
    { day: 14, week: 2, topic: 'Week 2 Review', sub: 'Mini Project: Data Wrangling', concepts: ['NumPy & Pandas Application'], time: 3, resource: 'https://www.kaggle.com/datasets', project: 'Find a simple dataset, load, clean, and prepare for analysis.' },

    // --- REVISED WEEK 3: Data Visualization & ML Intro (7 days) ---
    { day: 15, week: 3, topic: 'Data Visualization', sub: 'Matplotlib Basics', concepts: ['`plt.figure`, `plt.plot`', 'Labels, Titles, Legends', 'Saving plots'], time: 3, resource: 'https://www.youtube.com/watch?v=matplotlib-for-beginners', project: 'Create a simple line plot of dummy data.' },
    { day: 16, week: 3, topic: 'Data Visualization', sub: 'Matplotlib Plot Types', concepts: ['Bar charts', 'Histograms', 'Scatter plots'], time: 4, resource: 'https://www.youtube.com/watch?v=sO7tGny56n0&list=PLc2gB4S_m9D7g_L_m0F2u4Y04_2n5_0G_&index=1', project: 'Visualize different aspects using various plot types.' },
    { day: 17, week: 3, topic: 'Data Visualization', sub: 'Seaborn Introduction & Plots', concepts: ['Seaborn vs Matplotlib', 'Distribution plots', 'Categorical plots'], time: 3, resource: 'https://www.youtube.com/watch?v=GjYd66oQcBM&list=PLc2gB4S_m9D7g_L_m0F2u4Y04_2n5_0G_&index=2', project: 'Explore data distributions using Seaborn.' },
    { day: 18, week: 3, topic: 'Data Visualization', sub: 'Seaborn Advanced Plots', concepts: ['Relational plots (`relplot`)', 'Heatmaps', 'Pair plots'], time: 4, resource: 'https://www.youtube.com/watch?v=O_rXWjYt3y8&list=PLc2gB4S_m9D7g_L_m0F2u4Y04_2n5_0G_&index=3', project: 'Create correlation heatmap and pair plot for a dataset.' },
    { day: 19, week: 3, topic: 'ML Foundations', sub: 'Intro to ML & Regression', concepts: ['Supervised vs Unsupervised', 'Regression vs Classification', 'Linear Regression basics'], time: 4, resource: 'https://www.youtube.com/watch?v=UK-eLwK3Lio&list=PLc2gB4S_m9D7g_L_m0F2u4Y04_2n5_0G_&index=4', project: 'Build a simple linear regression model.' },
    { day: 20, week: 3, topic: 'ML Foundations', sub: 'Model Evaluation (Regression)', concepts: ['MAE, MSE, RMSE', 'R-squared', 'Train-test split'], time: 3, resource: 'https://www.youtube.com/watch?v=q6t8r6j8c9g&list=PLc2gB4S_m9D7g_L_m0F2u4Y04_2n5_0G_&index=5', project: 'Evaluate your first regression model.' },
    { day: 21, week: 3, topic: 'Week 3 Review', sub: 'EDA & First ML Model', concepts: ['Combining Pandas, Viz, & ML'], time: 4, resource: 'https://www.kaggle.com/learn/data-visualization', project: 'Perform a full EDA on a new dataset and prepare for ML.' },

    // --- REVISED WEEK 4: Classification & Unsupervised Learning (7 days) ---
    { day: 22, week: 4, topic: 'ML Foundations', sub: 'Logistic Regression', concepts: ['Classification theory', 'Sigmoid function', 'Scikit-learn implementation'], time: 4, resource: 'https://www.youtube.com/watch?v=yIYKR4sgzI8', project: 'Build a logistic regression model for binary classification.' },
    { day: 23, week: 4, topic: 'ML Foundations', sub: 'Model Evaluation (Classification)', concepts: ['Accuracy, Precision, Recall', 'F1-score', 'Confusion Matrix'], time: 4, resource: 'https://www.youtube.com/watch?v=Kdsp6soqA7o', project: 'Evaluate your classification model.' },
    { day: 24, week: 4, topic: 'Unsupervised Learning', sub: 'Clustering with K-Means', concepts: ['Clustering concept', 'K-Means algorithm', 'Elbow method'], time: 3, resource: 'https://www.youtube.com/watch?v=4b5d3muwG8Q', project: 'Apply K-Means clustering to a dataset.' },
    { day: 25, week: 4, topic: 'Feature Engineering', sub: 'Data Preprocessing', concepts: ['Scaling (StandardScaler, MinMaxScaler)', 'Encoding categorical variables'], time: 3, resource: 'https://www.youtube.com/watch?v=0B5eIE_1vpU', project: 'Preprocess a dataset for ML.' },
    { day: 26, week: 4, topic: 'Advanced ML', sub: 'Decision Trees', concepts: ['Tree structure', 'Gini impurity', 'Overfitting'], time: 4, resource: 'https://www.youtube.com/watch?v=ZVR2Way4nwQ', project: 'Train and visualize a decision tree classifier.' },
    { day: 27, week: 4, topic: 'Advanced ML', sub: 'Random Forests', concepts: ['Ensemble learning', 'Bagging', 'Feature importance'], time: 4, resource: 'https://www.youtube.com/watch?v=J4Wdy0Wc_xQ', project: 'Build a Random Forest model.' },
    { day: 28, week: 4, topic: 'Week 4 Review', sub: 'Classification Project', concepts: ['End-to-end ML classification pipeline'], time: 4, resource: 'https://www.kaggle.com/c/titanic', project: 'Work on the Titanic survival prediction problem.' },

    // --- REVISED WEEK 5: Advanced ML & Time Series (7 days) ---
    { day: 29, week: 5, topic: 'Advanced ML', sub: 'Support Vector Machines (SVM)', concepts: ['Maximal margin classifier', 'Support vectors', 'Kernels (linear, RBF)'], time: 4, resource: 'https://www.youtube.com/watch?v=_PwhiWxHK8o', project: 'Apply SVM to a classification problem.' },
    { day: 30, week: 5, topic: 'Advanced ML', sub: 'Gradient Boosting (XGBoost)', concepts: ['Boosting concept', 'XGBoost library', 'Parameter tuning'], time: 4, resource: 'https://www.youtube.com/watch?v=8b1JEDvenQU', project: 'Use XGBoost for a classification or regression task.' },
    { day: 31, week: 5, topic: 'Advanced ML', sub: 'Hyperparameter Tuning & Cross-Validation', concepts: ['GridSearchCV', 'RandomizedSearchCV', 'Cross-validation'], time: 3, resource: 'https://www.youtube.com/watch?v=H6y_i_f2trw', project: 'Tune hyperparameters of an ML model.' },
    { day: 32, week: 5, topic: 'Time Series', sub: 'Time Series Basics', concepts: ['Trends, Seasonality, Cyclicity', 'Stationarity', 'Autocorrelation (ACF, PACF)'], time: 4, resource: 'https://www.youtube.com/watch?v=v5-l3D4O-h4', project: 'Decompose a time series dataset.' },
    { day: 33, week: 5, topic: 'Time Series', sub: 'ARIMA Models', concepts: ['Autoregressive (AR)', 'Moving Average (MA)', 'Building an ARIMA model'], time: 4, resource: 'https://www.youtube.com/watch?v=e8Yw4alG16Q', project: 'Forecast future values of a time series dataset.' },
    { day: 34, week: 5, topic: 'Core Libraries', sub: 'SciPy & Statsmodels', concepts: ['Intro to SciPy (stats)', 'Intro to Statsmodels (OLS)'], time: 3, resource: 'https://www.scipy.org/getting-started.html', project: 'Perform a t-test on a sample dataset.' },
    { day: 35, week: 5, topic: 'Week 5 Review', sub: 'Advanced ML & Time Series Project', concepts: ['Applying advanced models'], time: 4, resource: 'https://www.kaggle.com/competitions', project: 'Participate in a beginner-friendly Kaggle competition.' },

    // --- REVISED WEEK 6: ML for Text & Image Data (7 days) ---
    { day: 36, week: 6, topic: 'ML for Text', sub: 'Bag-of-Words & TF-IDF', concepts: ['Text preprocessing (tokenization, stop words)', 'CountVectorizer', 'TfidfVectorizer'], time: 4, resource: 'https://www.youtube.com/watch?v=f_n11D-t30I', project: 'Build a simple spam classifier using TF-IDF.' },
    { day: 37, week: 6, topic: 'ML for Text', sub: 'Sentiment Analysis', concepts: ['Using ML for sentiment classification', 'VADER library', 'TextBlob library'], time: 3, resource: 'https://www.youtube.com/watch?v=QpzE7f2-g-c', project: 'Perform sentiment analysis on product reviews.' },
    { day: 38, week: 6, topic: 'ML for Images', sub: 'Intro to Image Data', concepts: ['Images as arrays (pixels)', 'Working with Pillow/OpenCV', 'Basic image manipulations'], time: 4, resource: 'https://www.youtube.com/watch?v=oXlwWbU8l2o', project: 'Load, display, and perform basic transformations on an image.' },
    { day: 39, week: 6, topic: 'ML for Images', sub: 'Image Classification with ML', concepts: ['Flattening images for ML models', 'Using SVM/Random Forest on image data', 'Limitations'], time: 4, resource: 'https://www.geeksforGeeks.org/image-classification-using-svm/', project: 'Classify images from a simple dataset (e.g., digits).' },
    { day: 40, week: 6, topic: 'Deep Learning', sub: 'Intro to Neural Networks', concepts: ['Perceptron', 'Activation functions (ReLU, Sigmoid)', 'Forward propagation'], time: 4, resource: 'https://www.youtube.com/watch?v=aircAruvnKk', project: 'Build a simple neural network from scratch using NumPy.' },
    { day: 41, week: 6, topic: 'Deep Learning', sub: 'Intro to Keras/TensorFlow', concepts: ['Sequential API', 'Dense layers', 'Compiling a model'], time: 4, resource: 'https://www.youtube.com/watch?v=Klv9C_2v1jI', project: 'Build a simple classifier using Keras.' },
    { day: 42, week: 6, topic: 'Week 6 Review', sub: 'Consolidate Skills', concepts: ['Review all topics, Work on a challenging project'], time: 4, resource: '', project: 'Choose a project that combines multiple skills (e.g., text analysis and visualization).' },

    // --- REVISED WEEK 7: Deep Learning - CNNs & Transfer Learning (7 days) ---
    { day: 43, week: 7, topic: 'Deep Learning', sub: 'Training & Evaluation Basics', concepts: ['Loss functions', 'Optimizers (Adam, SGD)', 'Epochs, Batch size'], time: 4, resource: 'https://www.youtube.com/watch?v=Kds2_3zF6eY&list=PLc2gB4S_m9D7g_L_m0F2u4Y04_2n5_0G_&index=6', project: 'Train and evaluate your Keras model.' },
    { day: 44, week: 7, topic: 'Deep Learning', sub: 'Convolutional Neural Networks (CNN) Architecture', concepts: ['Convolutional layers', 'Pooling layers', 'CNN architecture'], time: 4, resource: 'https://www.youtube.com/watch?v=FTr3n7uBIuE', project: 'Design a simple CNN architecture.' },
    { day: 45, week: 7, topic: 'Deep Learning', sub: 'Training a CNN', concepts: ['Data augmentation', 'Callbacks (EarlyStopping)', 'Evaluating CNN performance'], time: 4, resource: 'https://www.tensorflow.org/tutorials/images/classification', project: 'Train your CNN on an image dataset (e.g., MNIST/CIFAR-10).' },
    { day: 46, week: 7, topic: 'Deep Learning', sub: 'Transfer Learning for Images', concepts: ['Pre-trained models (VGG, ResNet)', 'Feature extraction', 'Fine-tuning'], time: 4, resource: 'https://www.youtube.com/watch?v=yJg-Y5byMMw', project: 'Use a pre-trained model to classify images on a new dataset.' },
    { day: 47, week: 7, topic: 'Deep Learning', sub: 'Recurrent Neural Networks (RNN) Basics', concepts: ['Handling sequential data', 'SimpleRNN, LSTM, GRU layers'], time: 4, resource: 'https://www.youtube.com/watch?v=Gl2WXLIMvKA', project: 'Implement a basic RNN for sequence data.' },
    { day: 48, week: 7, topic: 'Deep Learning', sub: 'Transfer Learning for Text (Embeddings)', concepts: ['Word Embeddings (Word2Vec, GloVe)', 'Using pre-trained embeddings'], time: 4, resource: 'https://www.youtube.com/watch?v=rIPJ9jXp_Xw&list=PLc2gB4S_m9D7g_L_m0F2u4Y04_2n5_0G_&index=8', project: 'Use pre-trained embeddings in an NLP model.' },
    { day: 49, week: 7, topic: 'Week 7 Review', sub: 'DL Image/Text Project', concepts: ['End-to-end DL project'], time: 4, resource: 'https://www.kaggle.com/c/dogs-vs-cats', project: 'Build a dog vs. cat classifier or sentiment analysis with DL.' },

    // --- REVISED WEEK 8: Chatbots & Portfolio Building (7 days) ---
    { day: 50, week: 8, topic: 'Deep Learning', sub: 'Intro to Chatbots (Rasa)', concepts: ['NLU, Core, Actions', 'Intents and Entities', 'Building a simple Rasa bot'], time: 5, resource: 'https://www.youtube.com/watch?v=j_2A-tM-o-k', project: 'Create a basic FAQ chatbot.' },
    { day: 51, week: 8, topic: 'Portfolio Building', sub: 'Project 1 - EDA', concepts: ['Select a dataset', 'Perform comprehensive EDA', 'Document findings clearly'], time: 5, resource: 'https://www.kaggle.com/datasets', project: 'Finalize a detailed EDA project for your portfolio.' },
    { day: 52, week: 8, topic: 'Portfolio Building', sub: 'Project 2 - ML', concepts: ['Select a regression/classification task', 'Build, evaluate, tune models', 'Document the entire process'], time: 5, resource: 'https://www.kaggle.com/competitions', project: 'Finalize a complete machine learning project for your portfolio.' },
    { day: 53, week: 8, topic: 'Portfolio Building', sub: 'Project 3 - DL (Optional)', concepts: ['Choose an image or text problem', 'Implement a deep learning solution', 'Showcase results'], time: 5, resource: '', project: 'Finalize a deep learning project for your portfolio.' },
    { day: 54, week: 8, topic: 'The Road Ahead', sub: 'Cloud & Big Data Intro', concepts: ['Intro to AWS/GCP/Azure for ML', 'What is Spark?'], time: 3, resource: 'https://www.youtube.com/watch?v=H6y_i_f2trw', project: 'Research cloud platforms for ML.' },
    { day: 55, week: 8, topic: 'The Road Ahead', sub: 'Advanced Topics & MLOps', concepts: ['Explainable AI (XAI)', 'Reinforcement Learning', 'Intro to MLOps'], time: 3, resource: 'https://www.youtube.com/watch?v=JgvyzIkg_zE', project: 'Read articles on advanced ML topics & MLOps.' },
    { day: 56, week: 8, topic: 'Week 8 Review', sub: 'GitHub & Resume Prep', concepts: ['Organizing projects on GitHub', 'Writing project READMEs', 'Updating your resume'], time: 4, resource: 'https://www.youtube.com/watch?v=RGOj5yH7evk', project: 'Create a professional GitHub profile and update resume.' },

    // --- REVISED WEEK 9: Final Review & Next Steps (4 days Remaining from 60) ---
    { day: 57, week: 9, topic: 'Final Prep', sub: 'Interview Questions', concepts: ['Common data analyst questions', 'SQL/Python coding challenges'], time: 4, resource: 'https://www.stratascratch.com/blog/data-analyst-interview-questions/', project: 'Practice interview questions and coding challenges.' },
    { day: 58, week: 9, topic: 'Final Prep', sub: 'Case Studies & Soft Skills', concepts: ['Solving data analysis case studies', 'Communication & Presentation skills'], time: 4, resource: 'https://www.mckinsey.com/capabilities/quantumblack/our-insights', project: 'Work through a data analytics case study.' },
    { day: 59, week: 9, topic: 'Portfolio Refinement', sub: 'Project Polish & Presentation', concepts: ['Refine portfolio projects', 'Prepare project presentations'], time: 4, resource: '', project: 'Ensure all portfolio projects are polished and presentable.' },
    { day: 60, week: 9, topic: 'Roadmap Completion!', sub: 'Celebrate & Plan Next Steps', concepts: ['Reflect on journey', 'Networking on LinkedIn', 'Continuous learning strategies'], time: 4, resource: '', project: 'Write a blog post about your 60-day learning journey and next goals.' },
];

// Motivational messages for daily boosts
const motivationalQuotes = [
    "You're not just learning, you're leveling up! Keep that grind strong.",
    "Every line of code is a step closer to data mastery. You're a coding legend!",
    "Don't stop when you're tired, stop when you're done. But also, take breaks!",
    "Data is the new oil, and you're learning to drill for it. Strike gold!",
    "Errors are just opportunities to learn. Debug it like a boss!",
    "Your brain is getting stronger with every new concept. Flex those neural networks!",
    "The future belongs to those who learn. You're building that future, one day at a time.",
    "Think of this as your personal data analytics bootcamp. You're crushing it!",
    "Small steps every day lead to big results. Consistency is key!",
    "You're literally turning raw data into insights. That's a superpower!",
    "Embrace the challenge, cherish the progress. You're doing amazing!"
];

// Project Recommendations (Level-wise)
const projectRecommendations = [
    {
        level: 'Beginner Projects (Weeks 1-3 Skills)',
        description: 'Perfect for solidifying Python basics, SQL, and initial data exploration/visualization.',
        projects: [
            { name: 'Basic E-commerce Data Analysis', details: 'Analyze sales data from a small e-commerce dataset (CSV). Clean data, calculate total sales, top products, customer distribution. Use Pandas, Matplotlib.', link: 'https://www.kaggle.com/datasets/carrieanne1776/ecommerce-data' },
            { name: 'Simple Weather Data Analysis', details: 'Explore a dataset of daily weather. Calculate averages, find patterns, visualize temperature trends. Use Pandas, Matplotlib/Seaborn.', link: 'https://www.kaggle.com/datasets/smid80/weather-dataset-from-weather-station' },
            { name: 'SQL Data Retrieval & Aggregation', details: 'Practice complex queries on a public SQL database (e.g., Chinook database). Use joins, group by, subqueries to answer business questions.', link: 'https://www.sqlitetutorial.net/sqlite-sample-database/' }
        ]
    },
    {
        level: 'Intermediate Projects (Weeks 4-6 Skills)',
        description: 'Step up your game with Machine Learning fundamentals for classification, regression, and working with text/time series.',
        projects: [
            { name: 'Predicting Customer Churn', details: 'Use a telecom customer dataset to predict if a customer will churn. Apply Logistic Regression or Decision Trees. Evaluate model performance. (Classification)', link: 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn' },
            { name: 'Housing Price Prediction', details: 'Predict house prices based on various features. Implement Linear Regression or Random Forests. Focus on feature engineering. (Regression)', link: 'https://www.kaggle.com/datasets/shivamss/house-price-prediction-challenge' },
            { name: 'Sentiment Analysis of Movie Reviews', details: 'Classify movie reviews as positive or negative. Use Bag-of-Words/TF-IDF and an ML classifier (e.g., Naive Bayes or SVM).', link: 'https://www.kaggle.com/datasets/lakshmi25npathi/sentiment-analysis-on-movie-reviews' }
        ]
    },
    {
        level: 'Advanced Projects (Weeks 7-9 Skills)',
        description: 'Dive deep into Neural Networks, Transfer Learning, and more complex data challenges to showcase advanced skills.',
        projects: [
            { name: 'Image Classification (Custom Dataset)', details: 'Build a CNN to classify images from a custom dataset (e.g., different types of flowers, animals). Experiment with data augmentation and fine-tuning pre-trained models.', link: 'https://www.kaggle.com/datasets/alessiocorrado99/animals10' },
            { name: 'Time Series Forecasting (Stock Prices/Sales)', details: 'Forecast stock prices or sales data using ARIMA/LSTM models. Focus on data stationarity and model tuning.', link: 'https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities' },
            { name: 'Build a Simple Chatbot', details: 'Using Rasa or a similar framework, create a goal-oriented chatbot that can answer FAQs or guide a user through a simple process.', link: 'https://rasa.com/docs/rasa/getting-started/your-first-assistant' }
        ]
    }
];

// All Learning Resources
const allLearningResources = [
    {
        category: 'YouTube Channels (Top Tier)',
        resources: [
            { name: 'Siddhardhan (Data Science & ML)', link: 'https://www.youtube.com/@siddhardhan-g/playlists' },
            { name: 'Krish Naik (Data Science & Deep Learning)', link: 'https://www.youtube.com/@krishnaik06/playlists' },
            { name: 'freeCodeCamp.org', link: 'https://www.youtube.com/@freecodecamp' },
            { name: 'Corey Schafer (Python Tutorials)', link: 'https://www.youtube.com/@coreyms' }
        ]
    },
    {
        category: 'Online Courses & Tutorials',
        resources: [
            { name: 'Andrew Ng - Machine Learning (Coursera)', link: 'https://www.coursera.org/learn/machine-learning' },
            { name: 'Kaggle Learn', link: 'https://www.kaggle.com/learn' },
            { name: 'Real Python', link: 'https://realpython.com/' },
            { name: 'W3Schools (SQL, HTML, CSS, JS)', link: 'https://www.w3schools.com/' }
        ]
    },
    {
        category: 'Coding Practice & Interview Prep',
        resources: [
            { name: 'HackerRank (Python, SQL)', link: 'https://www.hackerrank.com/' },
            { name: 'StrataScratch (SQL, Python Interview Questions)', link: 'https://www.stratascratch.com/' },
            { name: 'LeetCode (Coding Challenges)', link: 'https://leetcode.com/' }
        ]
    },
    {
        category: 'Data Sources & Datasets',
        resources: [
            { name: 'Kaggle Datasets', link: 'https://www.kaggle.com/datasets' },
            { name: 'UCI Machine Learning Repository', link: 'https://archive.ics.uci.edu/ml/index.php' },
            { name: 'Google Dataset Search', link: 'https://datasetsearch.research.google.com/' }
        ]
    }
];


// Get DOM elements
const mainTabsContainer = document.getElementById('main-tabs');
const dashboardContent = document.getElementById('dashboard-content');
const weeklyTasksContent = document.getElementById('weekly-tasks-content');
const projectsContent = document.getElementById('projects-content');
const resourcesContent = document.getElementById('resources-content');

const weekTabsContainer = document.getElementById('week-tabs');
const dailyCardsContainer = document.getElementById('daily-cards-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const motivationQuoteElement = document.getElementById('motivation-quote');
const confettiContainer = document.getElementById('confetti-container');
const projectListContainer = document.getElementById('project-list');
const resourceListContainer = document.getElementById('resource-list');
let weeklyHoursChart = null; // To store the Chart.js instance

// --- Local Storage Functions ---
// Function to save the completed days to local storage
function saveData() {
    localStorage.setItem('dataAnalyticsRoadmapProgress', JSON.stringify(Array.from(appState.completedDays)));
}

// Function to load saved progress from local storage
function loadData() {
    const savedData = localStorage.getItem('dataAnalyticsRoadmapProgress');
    if (savedData) {
        appState.completedDays = new Set(JSON.parse(savedData));
    }
}

// --- App State ---
const appState = {
    currentMainTab: 'dashboard', // Default active main tab
    currentWeek: 1, // Default to Week 1 for the Weekly Tasks tab
    completedDays: new Set(), // Store completed day numbers
};

// --- Chart Functions ---
// Get all unique week numbers from the roadmap data
function getWeeks() {
    return [...new Set(roadmapData.map(day => day.week))].sort((a, b) => a - b);
}

// Calculate total estimated learning hours for each week
function calculateWeeklyHours() {
    const weeks = getWeeks();
    const weeklyData = weeks.map(week => {
        const totalHours = roadmapData
            .filter(day => day.week === week)
            .reduce((sum, day) => sum + day.time, 0);
        return { week, totalHours };
    });
    return weeklyData;
}

// Create or update the Chart.js bar chart
function createChart() {
    const weeklyData = calculateWeeklyHours();
    const ctx = document.getElementById('weeklyHoursChart').getContext('2d');

    // Destroy previous chart instance if it exists to prevent overlap
    if (weeklyHoursChart) {
        weeklyHoursChart.destroy();
    }

    weeklyHoursChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: weeklyData.map(d => `Week ${d.week}`),
            datasets: [{
                label: 'Estimated Hours',
                data: weeklyData.map(d => d.totalHours),
                backgroundColor: 'var(--color-accent-green)', /* Now green! */
                borderColor: 'var(--color-accent-green)',
                borderWidth: 1,
                borderRadius: 10, /* More rounded bars */
                hoverBackgroundColor: 'var(--color-accent-blue)' /* Blue for hover */
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Allows chart to fill container without fixed aspect ratio
            plugins: {
                title: {
                    display: true,
                    text: 'Estimated Learning Hours per Week',
                    font: { size: 20, weight: '700', family: 'Inter' }, /* Match body font, larger */
                    color: 'var(--color-text-light)', /* Use light text color */
                    padding: { bottom: 25 }
                },
                legend: {
                    display: false // Hide legend as there's only one dataset
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw} hours`;
                        }
                    },
                    backgroundColor: 'var(--color-bg-secondary)',
                    titleColor: 'var(--color-text-light)',
                    bodyColor: 'var(--color-text-medium)',
                    titleFont: { family: 'Inter', size: 15, weight: '600' },
                    bodyFont: { family: 'Inter', size: 14 },
                    padding: 12,
                    cornerRadius: 8,
                    borderColor: 'var(--color-border-dark)',
                    borderWidth: 1,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Hours',
                        color: 'var(--color-text-medium)',
                        font: { family: 'Inter', size: 15 }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.08)', /* Lighter subtle grid lines on dark background */
                        drawBorder: false, /* Remove y-axis border */
                    },
                    ticks: {
                        color: 'var(--color-text-dark)', /* Tick labels color */
                        font: { family: 'Inter', size: 13 }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Week',
                        color: 'var(--color-text-medium)',
                        font: { family: 'Inter', size: 15 }
                    },
                    grid: {
                        display: false /* No vertical grid lines */
                    },
                    ticks: {
                        color: 'var(--color-text-dark)', /* Tick labels color */
                        font: { family: 'Inter', size: 13 }
                    }
                }
            }
        }
    });
}

// --- Render Functions ---
// Renders the main tab navigation buttons and controls content visibility
function renderMainTabs() {
    const mainTabButtons = document.querySelectorAll('.main-tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    mainTabButtons.forEach(button => {
        const tabName = button.dataset.tab;
        // Reset button styles
        button.classList.remove('main-tab-button-active');
        button.classList.add('bg-secondary', 'text-medium', 'border-dark', 'shadow-advillains');

        // Set active style for the current tab
        if (tabName === appState.currentMainTab) {
            button.classList.add('main-tab-button-active');
        }
    });

    // Hide all tab content sections
    tabContents.forEach(content => {
        content.classList.add('hidden');
    });

    // Show the active tab content section
    document.getElementById(`${appState.currentMainTab}-content`).classList.remove('hidden');

    // Re-render specific sections when their respective main tab is active
    if (appState.currentMainTab === 'weekly-tasks') {
        renderNavigation(); // Render weekly navigation if weekly tasks tab is active
        renderDailyCards();
    } else if (appState.currentMainTab === 'projects') {
        renderProjectRecommendations();
    } else if (appState.currentMainTab === 'resources') {
        renderAllResources();
    } else if (appState.currentMainTab === 'dashboard') {
        updateDashboard();
    }
}

// Renders the weekly navigation buttons (within the Weekly Tasks tab)
function renderNavigation() {
    weekTabsContainer.innerHTML = ''; // Clear previous buttons
    const weeks = getWeeks();
    weeks.forEach(week => {
        const button = document.createElement('button');
        button.textContent = `Week ${week}`;
        // Apply Tailwind classes for styling and hover effects, using CSS variables
        button.className = `nav-button px-6 py-3 text-base font-medium rounded-lg transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent-blue`;

        // Add active class if it's the current week
        if (week === appState.currentWeek) {
            button.classList.add('nav-button-active');
        }

        // Event listener to switch weeks when a button is clicked
        button.addEventListener('click', () => {
            appState.currentWeek = week;
            renderNavigation(); // Re-render to update active state
            renderDailyCards(); // Render cards for the newly selected week
        });
        weekTabsContainer.appendChild(button);
    });
}

// Renders the daily learning cards for the currently selected week
function renderDailyCards() {
    dailyCardsContainer.innerHTML = ''; // Clear previous cards
    const daysForWeek = roadmapData.filter(day => day.week === appState.currentWeek);

    daysForWeek.forEach(day => {
        const isCompleted = appState.completedDays.has(day.day);
        const card = document.createElement('div');
        // Apply Tailwind classes for card styling and completed state, using CSS variables
        card.className = `day-card p-6 bg-secondary rounded-xl shadow-advillains ${isCompleted ? 'completed-card' : ''}`;

        card.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <p class="text-sm font-medium text-text-medium">Day ${day.day}</p>
                    <h3 class="text-2xl font-bold mt-1 text-light">${day.topic}</h3>
                    <p class="text-md text-accent-blue font-semibold">${day.sub}</p>
                </div>
                <div class="flex items-center">
                    <input type="checkbox" id="day-${day.day}" data-day="${day.day}" class="appearance-none h-6 w-6 rounded-md border-2 border-text-medium checked:border-accent-green focus:ring-2 focus:ring-accent-blue focus:ring-offset-2 transition-all duration-200" ${isCompleted ? 'checked' : ''}>
                </div>
            </div>
            <div class="mt-4">
                <p class="font-semibold text-light">Key Concepts:</p>
                <ul class="list-disc list-inside text-text-medium text-sm mt-1 space-y-1 pl-4">
                    ${day.concepts.map(concept => `<li>${concept}</li>`).join('')}
                </ul>
            </div>
            <div class="mt-4">
                <p class="font-semibold text-light">Today's Project:</p>
                <p class="text-text-medium text-sm mt-1">${day.project}</p>
            </div>
            <div class="mt-5 pt-4 border-t border-border-dark flex justify-between items-center">
                <span class="text-sm font-medium text-text-medium">🕒 ${day.time} hours</span>
                ${day.resource ? `<a href="${day.resource}" target="_blank" rel="noopener noreferrer" class="text-sm font-semibold text-accent-blue hover:text-accent-primary-highlight transition-colors flex items-center">
                    <svg class="icon-sm mr-1" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V3a1 1 0 00-1-1h-6z"></path><path d="M4 10a2 2 0 00-2 2v5a2 2 0 002 2h5a2 2 0 002-2v-2a1 1 0 00-1-1H7a1 1 0 01-1-1v-2a1 1 0 00-1-1H4z"></path></svg>
                    Learning Resource
                </a>` : `<span class="text-sm font-medium text-text-dark opacity-70">No Resource Yet!</span>`}
            </div>
        `;
        dailyCardsContainer.appendChild(card);
    });
}

// Renders the project recommendations section
function renderProjectRecommendations() {
    projectListContainer.innerHTML = ''; // Clear previous content
    projectRecommendations.forEach(section => {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'p-8 bg-primary rounded-xl shadow-advillains border border-border-dark'; /* Primary background for inner section */
        sectionDiv.innerHTML = `
            <h3 class="text-3xl font-bold text-accent-primary-highlight mb-5">${section.level}</h3>
            <p class="text-medium text-base mb-7">${section.description}</p>
            <ul class="space-y-5">
                ${section.projects.map(p => `
                    <li class="border-b border-border-dark pb-5 last:border-b-0 last:pb-0">
                        <a href="${p.link}" target="_blank" rel="noopener noreferrer" class="text-light hover:text-accent-yellow font-semibold text-xl block transition-colors flex items-center">
                            <svg class="icon-sm mr-2 text-accent-blue" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M10.293 15.707a1 1 0 010-1.414L14.586 10l-4.293-4.293a1 1 0 111.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z" clip-rule="evenodd"></path><path fill-rule="evenodd" d="M4.293 15.707a1 1 0 010-1.414L8.586 10l-4.293-4.293a1 1 0 111.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z" clip-rule="evenodd"></path></svg>
                            ${p.name}
                        </a>
                        <p class="text-text-medium text-sm mt-2 ml-7">${p.details}</p>
                    </li>
                `).join('')}
            </ul>
        `;
        projectListContainer.appendChild(sectionDiv);
    });
}

// Renders the all learning resources section
function renderAllResources() {
    resourceListContainer.innerHTML = ''; // Clear previous content
    allLearningResources.forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'resource-box p-8 bg-secondary rounded-xl shadow-advillains border border-border-dark';
        categoryDiv.innerHTML = `
            <h3 class="text-2xl font-bold text-light mb-5">${category.category}</h3>
            <ul class="space-y-4">
                ${category.resources.map(res => `
                    <li>
                        <a href="${res.link}" target="_blank" rel="noopener noreferrer" class="text-accent-blue hover:text-accent-primary-highlight font-semibold text-base block transition-colors flex items-center">
                            <svg class="icon-sm mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V3a1 1 0 00-1-1h-6z"></path><path d="M4 10a2 2 0 00-2 2v5a2 2 0 002 2h5a2 2 0 002-2v-2a1 1 0 00-1-1H7a1 1 0 01-1-1v-2a1 1 0 00-1-1H4z"></path></svg>
                            ${res.name}
                        </a>
                    </li>
                `).join('')}
            </ul>
        `;
        resourceListContainer.appendChild(categoryDiv);
    });
}

// Updates the progress bar and text, and displays a random motivational quote
function updateDashboard() {
    const completedCount = appState.completedDays.size;
    const totalDays = roadmapData.length;
    const progressPercentage = (completedCount / totalDays) * 100;

    // Update progress bar width and text
    progressBar.style.width = `${progressPercentage}%`;
    progressText.textContent = `${completedCount} / ${totalDays} Unlocked`;

    // Update motivational quote randomly
    const randomQuote = motivationalQuotes[Math.floor(Math.random() * motivationalQuotes.length)];
    // Apply text gradient to the quote span
    motivationQuoteElement.querySelector('span').innerHTML = `
        <span style="background: linear-gradient(45deg, var(--color-accent-blue), var(--color-accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent; font-weight: 600;">
            "${randomQuote}"
        </span>
    `;
}

// --- Confetti Effect ---
// Creates a single confetti particle with random properties
function createConfetti(x, y, color) {
    const confetti = document.createElement('div');
    confetti.className = 'confetti';
    confetti.style.left = `${x}px`;
    confetti.style.top = `${y}px`;
    confetti.style.backgroundColor = color;
    confetti.style.setProperty('--dx', `${(Math.random() - 0.5) * 300}px`); /* Wider spread */
    confetti.style.setProperty('--dy', `${(Math.random() * 250) - 300}px`); /* Higher launch */
    confetti.style.setProperty('--rot', `${Math.random() * 1080}deg`); /* More rotation */

    confettiContainer.appendChild(confetti);

    // Remove confetti after animation to prevent DOM bloat
    confetti.addEventListener('animationend', () => {
        confetti.remove();
    });
}

// Triggers a burst of confetti when a checkbox is checked
function triggerConfetti(e) {
    if (e.target.matches('input[type="checkbox"]') && e.target.checked) {
        const rect = e.target.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        // Confetti colors matching the new theme accents
        const colors = ['var(--color-accent-primary-highlight)', 'var(--color-accent-blue)', 'var(--color-accent-green)', 'var(--color-accent-purple)', 'var(--color-accent-yellow)'];

        for (let i = 0; i < 40; i++) { /* Even more confetti particles */
            const randomColor = colors[Math.floor(Math.random() * colors.length)];
            createConfetti(centerX + (Math.random() - 0.5) * 50, centerY + (Math.random() - 0.5) * 50, randomColor);
        }
    }
}

// --- Event Handlers ---
// Handles changes to checkboxes (marking days complete/incomplete)
function handleProgressToggle(e) {
    // Listen for changes on checkboxes within the daily cards container using event delegation
    if (e.target.matches('input[type="checkbox"]')) {
        const dayNumber = parseInt(e.target.dataset.day, 10);
        if (e.target.checked) {
            appState.completedDays.add(dayNumber);
            triggerConfetti(e); // Trigger confetti on completion
        } else {
            appState.completedDays.delete(dayNumber);
        }
        saveData(); // Save updated progress to local storage
        updateDashboard(); // Update dashboard visualization (progress bar, quote)
        // Toggle 'completed-card' class on the parent card for visual feedback
        e.target.closest('.day-card').classList.toggle('completed-card', e.target.checked);
    }
}

// --- Initialization ---
// Function to run when the DOM is fully loaded
function initializeApp() {
    loadData(); // Load saved progress from local storage
    createChart(); // Initialize the weekly hours chart

    // Add event listener for main tabs (delegation)
    mainTabsContainer.addEventListener('click', (e) => {
        if (e.target.closest('.main-tab-button')) { /* Use closest for better click area */
            appState.currentMainTab = e.target.closest('.main-tab-button').dataset.tab;
            renderMainTabs();
        }
    });

    // Initial rendering of main tabs and their content (defaults to dashboard)
    renderMainTabs();

    // Add event listener for daily task checkboxes (event delegation)
    dailyCardsContainer.addEventListener('change', handleProgressToggle);
}

// Initialize the app when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);
