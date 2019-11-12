import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from numpy import logspace, exp, log
import statsmodels.api as sm


def test_columns(df, df_test):
    for column in df.columns:
        if (column not in df_test.columns) & (column != 'saleprice')& (column != 'logsaleprice'):
            df_test[column] = 0
            print(f'Added {column} to the test dataframe')



def do_the_model(df,
                 features,
                 target,
                 scaled=False,
                 random_state=1,
                 model_type=LinearRegression
                ):


    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = \
    train_test_split(X,y,
                     random_state=random_state,
                    )

    if scaled==True:
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)

    if model_type == LassoCV or model_type == RidgeCV or model_type == ElasticNetCV:
        alphas = logspace(-5,5,500)

    if model_type == LassoCV or model_type == ElasticNetCV:
        model = model_type(alphas = alphas,
               cv=5,
               max_iter=5000,
              )
    elif model_type == RidgeCV:
        model = model_type(alphas = alphas,
               cv=5,
              )

    else:
        model = model_type()
    model.fit(X_train,y_train)
    print(f'Training Data Score: {model.score(X_train,y_train)}')
    print(f'Test Data Score: {model.score(X_test,y_test)}')
    print(f'Cross Validation Score: {cross_val_score(model, X_train, y_train, cv=5).mean()}')
    return([model,
    model.score(X_train,y_train),
    model.score(X_test,y_test),
    cross_val_score(model, X_train, y_train, cv=5).mean()
    ])

def new_test(df,
             test_df,
             features,
             target,
             random_state,
             model_type,
            model_df,
            scaled,
            ):

    test_columns(df=df, df_test=test_df)

    #make the model
    new_model = do_the_model(df=df,
                            features=features,
                            target=target,
                            random_state=random_state,
                            model_type=model_type,
                            scaled=scaled
                                     )

    try:
        model_id = model_df['model_id'].max() + 1
    except:
        model_id = 1
    if target == 'saleprice':
        test_df['SalePrice'] = new_model[0].predict(test_df[features])
    elif target == 'logsaleprice':
        test_df['SalePrice'] = new_model[0].predict(test_df[features])
        test_df['SalePrice'] = exp(test_df['SalePrice'])
    else:
        pass



    test_df[['id','SalePrice']].to_csv('kaggle/submission'+str(model_id)+'.csv', index=False)
    try:
        alpha = new_model[0].alpha_
    except:
        alpha = 'model has none'
    model_dict = {
        'model_id': model_id,
        'features_used': features,
        'target': target,
        'random_state': random_state,
        'model_type': model_type,
        'random_state': random_state,
        'scaled':scaled,
        'alpha': alpha,
        'coeffs': new_model[0].coef_,
        'intercept':new_model[0].intercept_,
        'train_score': new_model[1],
        'test_score': new_model[2],
        'cross_val_score': new_model[3],
        'output': 'submission'+str(int(model_id))+'.csv',

    }

    new_df = model_df.append(model_dict, ignore_index=True)
    new_df.to_csv('datasets/models.csv', index=False)
    return(new_df)

def model_maker(data,
                potential_features,
                target,
                model_type=LinearRegression,
                random_states=[1],
               ):
    new_df_columns = ['model_id',
                'features_included',
                'target',
                'random_state',
                'train_score',
                'test_score',
                'cross_val_score']
    new_df = pd.DataFrame(columns = new_df_columns)




    combos_possible = []
    from itertools import combinations
    for i in range(1,len(potential_features)+1):
        combos = list(combinations(potential_features, i))
        for combo in combos:
            combos_possible.append(list(combo))


    for model_id in range(len(combos_possible)):
        features = combos_possible[model_id]

        for random_state in random_states:
            model = model_type()
            X = data[features]
            type(X)
            y = data[target]
            X_train, X_test, y_train, y_test = \
            train_test_split(X,y,
                             random_state=random_state,
                            )
            model.fit(X_train,y_train)

            model_dict = {
                'model_id': model_id,
                'features_included': features,
                'target': target,
                'random_state': random_state,
                'train_score': model.score(X_train,y_train),
                'test_score': model.score(X_test,y_test),
                'cross_val_score': cross_val_score(model, X_train, y_train, cv=5).mean()
                }
            new_df = new_df.append(model_dict, ignore_index=True)
    return(new_df)
