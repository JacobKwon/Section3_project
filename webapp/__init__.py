import string
from flask import Flask
from flask import Flask, request, redirect, render_template
import pandas as pd

from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import plotly.express as px
from dash.dependencies import Input, Output
from dash import dash_table

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import base64
from io import BytesIO

import pymysql

con = pymysql.connect(
    host='',
    user='',
    password='',
    db='',
    charset='utf8mb4', 
    autocommit=True, 
    cursorclass=pymysql.cursors.DictCursor 
)
cur = con.cursor()

sql = "select trending_date, count(trending_date) AS data_cnt from youtube_data group by trending_date"
cur.execute(sql)
rows = cur.fetchall()
df = pd.DataFrame(rows)
fig = px.bar(df,x='trending_date',y='data_cnt')

sql = "SELECT category_id, count(category_id) AS category_cnt FROM youtube_data GROUP BY category_id"
cur.execute(sql)
rows = cur.fetchall()
df2 = pd.DataFrame(rows)
fig2 = px.pie(df2, values='category_cnt', names='category_id', title='Categorys')

con.close()

def create_app():
    
    app = Flask(__name__)
    dash_app = Dash(__name__, server = app, url_base_pathname='/dashboard/')

    dash_app.layout = html.Div([
        html.Div(id='img_wrap',
                className='img_wrap_class',
                style={'text-align': 'center'},
                children=[
                    html.Img(id="image_wc"),
                ]),
        html.Div(id='date_grp',
                className='date_grp_class',
                style={'margin': '10ex 0 10ex 0'},
                children=[
                    dcc.Graph(
                        id='date_dcc_grp',
                        figure=fig
                    )
                ]),
        html.Div(id='piecrt',
                className='pie_class',
                style={'margin': '10ex 0 10ex 0'},
                children=[
                    dcc.Graph(
                        id='pie_chart',
                        figure=fig2
                    )
                ]),
    ])

    def plot_wordcloud(data):
        d = {a: x for a, x in data.values}
        wc = WordCloud(background_color='black', width=480, height=360, font_path='/Library/Fonts/Arial Unicode.ttf')
        wc.fit_words(d)
        return wc.to_image()

    # @dash_app.callback(Output("date_dcc_grp", "figure"))
    # def update_bar_chart():
    #     cur = con.cursor()
    #     sql = "select trending_date, count(trending_date) AS data_cnt from youtube_data group by trending_date"
    #     cur.execute(sql)
    #     rows = cur.fetchall()
    #     df = pd.DataFrame(rows)
    #     # df
    #     con.close()
    #     print(df.head())
    #     # mask = (df['trending_date'] > low) & (df['data_cnt'] < high)
    #     fig = px.bar(df,x='trending_date',y='data_cnt')
    #     return fig


    @dash_app.callback(dd.Output('image_wc', 'src'), [dd.Input('image_wc', 'id')])
    def make_image(b):
        model = Word2Vec.load("/Users/jacob/Documents/CodeStates/Section_3/Sprint_3/webapp/word_model")
        cnt = Counter(model.wv.index_to_key).most_common(20)
        m_columns = ' '.join([str(v[0]) for v in cnt])
        m_columns = m_columns.split(" ")
        # m_columns
        k_df = pd.DataFrame(data=m_columns, columns=['Keyword'])
        k_df['Count'] = 0
        for idx, row in k_df.iterrows():
            k_df.loc[idx,'Count'] = model.wv.get_vecattr(row['Keyword'], "count")
        # k_df
        
        img = BytesIO()
        plot_wordcloud(data=k_df).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/result/', methods=['POST'])
    def show_keyword():
        if request.method == 'POST':
            res = request.form['keyword']

            if "__dash__" in res:
                return redirect('/dashboard/')

            res = res.replace(' ','')
            str_to_list = res.split(',')

            cnt = len(str_to_list)

            model_res_df = model_image(str_to_list)

            model_res_df.reset_index(inplace=True)
            model_res_list = []
            for i in range(0,cnt):
                model_res_list.append(model_res_df[i*5: ((i+1)*5)]['단어'].values.tolist())
            
        if res == ['']:
            return redirect('/')
        else:
            return render_template('result.html', result=res, cnt=cnt, listdata=model_res_list)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run()






#키워드->모델에서 검색할 함수
def model_image(lists):

    plt.switch_backend('Agg')#보여주지않고 이미지 저장 위한 세팅
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'AppleGothic'

    res_keyword = pd.DataFrame(columns=['단어'])

    model = Word2Vec.load("/urls/word_model")

    for i in range(0,len(lists)):
        # print(lists[i])
        tmp = pd.DataFrame(model.wv.most_similar(lists[i], topn=10), columns=['단어','유사도'])
        res_keyword = pd.concat([res_keyword, tmp.loc[[0,1,2,3,4],['단어']] ])

        sns.set_theme(style="whitegrid", font='AppleGothic')

        g = sns.PairGrid(tmp.sort_values("유사도", ascending=False),
                        x_vars=tmp.columns[1:], y_vars=["단어"],
                        height=7, aspect=.25)

        g.map(sns.stripplot, size=10, orient="h", jitter=False,
            palette="flare_r", linewidth=1, edgecolor="w"
            )

        #유사도 중 가장 낮은 값 찾아서 보기좋게 -0.1 해준다.
        min_val = tmp['유사도'].min() - 0.1
        g.set(xlim=(min_val, 1), xlabel="유사도", ylabel="")

        # titles = f'키워드 : {lists[i]}'
        titles=''

        for ax, title in zip(g.axes.flat, titles):
            ax.set(title=title)
            # ax.title.set_position([.1,1.1])
            # ax.title.set_text(fontsize=1)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True)

        sns.despine(left=True, bottom=True)
        plt.savefig(f'/urls/webapp/static/img/{i}.png')
    
    return res_keyword