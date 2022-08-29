import pandas as pd

def model_image(lists):
    from gensim.models import Word2Vec
    import matplotlib.pyplot as plt
    import seaborn as sns

    # import os
    # print(__file__)
    # print(os.path.realpath(__file__))
    # print(os.path.abspath(__file__))

    plt.switch_backend('Agg')#보여주지않고 이미지 저장 위한 세팅
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'AppleGothic'

    res_keyword = pd.DataFrame(columns=['단어'])

    model = Word2Vec.load("/Users/jacob/Documents/CodeStates/Section_3/Sprint_3/webapp/word_model")

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

        titles = ''

        for ax, title in zip(g.axes.flat, titles):
            ax.set(title=title)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True)

        sns.despine(left=True, bottom=True)
        # plt.savefig(f'/Users/jacob/Documents/CodeStates/Section_3/Sprint_3/webapp/{i}.png')
    
    return res_keyword


keywords = ['먹방','여행']

datas = model_image(keywords)
datas