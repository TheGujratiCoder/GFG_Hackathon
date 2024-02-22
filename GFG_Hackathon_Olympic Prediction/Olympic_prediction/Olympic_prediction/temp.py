
# def index(request):
#     # Check if the trained model results are cached
#     if not cache.get('model_results'):
#         # Read the medals data from the CSV file
#         df = pd.read_csv("./templates/medals.csv")

#         # Perform data preprocessing and generate necessary statistics
#         # ...
#         df=pd.read_csv("./medals.csv")
#         df1=df

#         del df['country_code']
#         del df['country_3_letter_code']
#         All_Athlete_URL=set()

#         for x in df['athlete_url']:
#             All_Athlete_URL.add(x)

#         del df['athlete_url']

#         df['athlete_full_name']=df.groupby(['discipline_title','medal_type'])['athlete_full_name'].transform(lambda x: ', '.join(str(i) for i in x))
#         df=df.drop_duplicates(subset=['discipline_title','slug_game','medal_type'])
#         df['athlete_full_name'].fillna('',inplace=True)

#         # df

#         # Train and evaluate models
#         svm_classification_report = train_svm(df)
#         random_forest_classification_report = train_random_forest(df)
#         xgboost_classification_report = train_xgboost(df)
#         mlp_classification_report = train_mlp(df)
#         lstm_classification_report = train_lstm(df)
#         # 4 Knowing the Names of all participants participated till now
#         l1=participant(df)
#         # Finding Number of Participants in All Sports
#         Sports_Types=Sports_Types(df)
#         # 5 Finding Number of Countries Participated

#         # Store the model results in the cache for future access
#         model_results = {
#             'l1':l1,
#             'Sports_Types':Sports_Types,
#             'svm_classification_report': svm_classification_report,
#             'random_forest_classification_report': random_forest_classification_report,
#             'xgboost_classification_report': xgboost_classification_report,
#             'mlp_classification_report': mlp_classification_report,
#             'lstm_classification_report': lstm_classification_report,
#         }
#         cache.set('model_results', model_results, timeout=None)  # Set timeout to None for no expiration

#     # Retrieve the cached model results
#     cached_model_results = cache.get('model_results')

#     # Prepare the data to pass to the template
#     content = {
#         'Sports_Types': cached_model_results[Sports_Types],
#         'l1': cached_model_results[l1],
#         'svm_classification_report': cached_model_results['svm_classification_report'],
#         'random_forest_classification_report': cached_model_results['random_forest_classification_report'],
#         'xgboost_classification_report': cached_model_results['xgboost_classification_report'],
#         'mlp_classification_report': cached_model_results['mlp_classification_report'],
#         'lstm_classification_report': cached_model_results['lstm_classification_report'],
#     }

#     return render(request, 'index.html', content)




# previous working


# def index(request):
#     # Check if the trained model results are cached
#     if not cache.get('model_results'):
#         # Read the medals data from the CSV file
#         # df = pd.read_csv("./templates/medals.csv")

#         # ...

#         # Train and evaluate models
#         # svm_classification_report = train_svm(df)
#         random_forest_classification_report = train_random_forest(df)
#         xgboost_classification_report = train_xgboost(df)
#         mlp_classification_report = train_mlp(df)
#         lstm_classification_report = train_lstm(df)
#         # l1 = participant(df)
#         # Sports_Types = get_sports_types(df)

#         # Store the model results in the cache for future access
#         model_results = {
#             # 'l1': l1,
#             # 'Sports_Types': Sports_Types,
#             # 'svm_classification_report': svm_classification_report,
#             'random_forest_classification_report': random_forest_classification_report,
#             'xgboost_classification_report': xgboost_classification_report,
#             'mlp_classification_report': mlp_classification_report,
#             'lstm_classification_report': lstm_classification_report,
#         }
#         cache.set('model_results', model_results, timeout=None)  # Set timeout to None for no expiration

#     # Retrieve the cached model results
#     cached_model_results = cache.get('model_results')

#     # Prepare the data to pass to the template
#     content = {
#         # 'Sports_Types': cached_model_results['Sports_Types'],
#         # 'l1': cached_model_results['l1'],
#         # 'svm_classification_report': cached_model_results['svm_classification_report'],
#         'random_forest_classification_report': cached_model_results['random_forest_classification_report'],
#         'xgboost_classification_report': cached_model_results['xgboost_classification_report'],
#         'mlp_classification_report': cached_model_results['mlp_classification_report'],
#         'lstm_classification_report': cached_model_results['lstm_classification_report'],
#     }

#     return render(request, 'index.html', content)

