import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import datetime as dt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from flask import Flask, render_template, request

app = Flask(__name__)


def convertToHtml(data):
    html = """
    <!DOCTYPE html>
    <html lang="en">

    <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-GLhlTQ8iRABdZLl6O3oVMWSktQOp6b7In1Zl3/Jr59b6EGGoI1aFkw7cmDA6j6gD" crossorigin="anonymous">
    </head>

    <body>
    """ + data + """

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js" integrity="sha384-w76AqPfDkMBDXo30jS1Sgez6pr3x5MlQ1ZAGC+nuZB+EYdgRZgiwxhTBTkF7CXvN" crossorigin="anonymous"></script>
    <script>
        window.onload = function(){
            document.getElementByTagName('table').classList.add('table')
        }
    </script>

    </body>

    </html>"""
    return html


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/cluster", methods=['POST', 'GET'])
def result():
    cluster_number = 2

    # validation if has request
    if request.method == 'POST':
        if request.form['cluster_number'] != '':
            cluster_number = int(request.form['cluster_number'])

    datahtml = ""
    # Load dataset
    df = pd.read_csv('OnlineRetail.csv', sep=",",
                     encoding="ISO-8859-1", header=0)

    # data cleaning

    datahtml += '<p>Jumlah data awal : ' + str(len(df)) + '</p>'
    df_null = round(100*(df.isnull().sum().to_frame())/len(df), 2)
    df = df.dropna()
    datahtml += '<p>Jumlah data hasil cleaning : ' + \
        str(len(df)) + '</p>' + "<br><h5>data</h5>" + df.head().to_html()
    df['CustomerID'] = df.CustomerID.astype(str)

    # Membuat atribut baru : Monetary
    df['Monetary'] = df['Quantity']*df['UnitPrice']
    rfm_m = df.groupby('CustomerID')['Monetary'].sum()
    rfm_m = rfm_m.reset_index()

    # Membuat atribut baru : Frequency
    rfm_f = df.groupby('CustomerID')['InvoiceNo'].count()
    rfm_f = rfm_f.reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']

    # merge atribut
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')

    df['InvoiceDate'] = pd.to_datetime(
        df['InvoiceDate'], format='%m/%d/%Y %H:%M')
    max_date = max(df['InvoiceDate'])
    df['Diff'] = max_date - df['InvoiceDate']

    rfm_p = df.groupby('CustomerID')['Diff'].min()
    rfm_p = rfm_p.reset_index()

    rfm_p['Diff'] = rfm_p['Diff'].dt.days

    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Monetary', 'Frequency', 'Recency']
    datahtml += "<br><h5>RFM Table</h5>" + rfm.head().to_html()

    # visualisasi data
    attributes = ['Monetary', 'Frequency', 'Recency']
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = [10, 8]
    sns.boxplot(data=rfm[attributes], orient="v",
                palette="Set2", whis=1.5, saturation=1, width=0.7)
    plt.title("Outliers Variable Distribution", fontsize=14, fontweight='bold')
    plt.ylabel("Range", fontweight='bold')
    plt.xlabel("Attributes", fontweight='bold')
    plt.savefig('static/result1.png')
    datahtml += '<img src="../static/result1.png" alt="result1">'
    plt.clf()

    # Removing (statistical) outliers for Monetary
    Q1 = rfm.Monetary.quantile(0.05)
    Q3 = rfm.Monetary.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Monetary >= Q1 - 1.5*IQR) & (rfm.Monetary <= Q3 + 1.5*IQR)]
    # Removing (statistical) outliers for Recency
    Q1 = rfm.Recency.quantile(0.05)
    Q3 = rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]
    # Removing (statistical) outliers for Frequency
    Q1 = rfm.Frequency.quantile(0.05)
    Q3 = rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) &
              (rfm.Frequency <= Q3 + 1.5*IQR)]

    # Rescaling Atribute
    rfm_df = rfm[['Monetary', 'Frequency', 'Recency']]
    # Instantiate
    scaler = StandardScaler()
    # fit_transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
    datahtml += '<p>hasil dari Standardisation Scaling</p>' + \
        rfm_df_scaled.head().to_html()

    # K-Means Clustering
    kmeans = KMeans(n_clusters=cluster_number, max_iter=50, n_init='auto')
    kmeans.fit(rfm_df_scaled)

    # Elbow-curve/SSD
    ssd = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50, n_init='auto')
        kmeans.fit(rfm_df_scaled)

        ssd.append(kmeans.inertia_)

    # plot the SSDs for each n_clusters
    plt.plot(ssd)
    plt.savefig('static/result2.png')
    datahtml += '<img src="../static/result2.png" alt="result2">'
    plt.clf()

    # Final model with k={value}
    kmeans = KMeans(n_clusters=cluster_number, max_iter=50, n_init='auto')
    kmeans.fit(rfm_df_scaled)

    # Assign the label
    rfm['Cluster_Id'] = kmeans.labels_
    datahtml += rfm.head().to_html()

    # Boxplot untuk memvisualisasikan Cluster Id dan Monetary
    sns.boxplot(x='Cluster_Id', y='Monetary', data=rfm)
    plt.savefig('static/result3.png')
    datahtml += '<img src="../static/result3.png" alt="result3">'
    plt.clf()

    # Boxplot untuk memvisualisasikan Cluster Id vs Frequency
    sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm)
    plt.savefig('static/result4.png')
    datahtml += '<img src="../static/result4.png" alt="result4">'
    plt.clf()

    # Boxplot untuk memvisualisasikan Cluster Id vs Recency
    sns.boxplot(x='Cluster_Id', y='Recency', data=rfm)
    plt.savefig('static/result5.png')
    datahtml += '<img src="../static/result5.png" alt="result5">'
    plt.clf()

    # Plotting Recency and monetary
    sns.scatterplot(x='Recency', y='Monetary', hue='Cluster_Id',
                    palette=sns.color_palette('hls', cluster_number), data=rfm, legend='full')
    plt.savefig('static/result6.png')
    datahtml += '<img src="../static/result6.png" alt="result6">'
    plt.clf()

    # Plotting Frequency and monetary
    sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster_Id',
                    palette=sns.color_palette('hls', cluster_number), data=rfm, legend='full')
    plt.savefig('static/result7.png')
    datahtml += '<img src="../static/result7.png" alt="result7">'
    plt.clf()

    # Plotting Frequency and Recency
    sns.scatterplot(x='Frequency', y='Recency', hue='Cluster_Id',
                    palette=sns.color_palette('hls', cluster_number), data=rfm, legend='full')
    plt.savefig('static/result8.png')
    datahtml += '<img src="../static/result8.png" alt="result8">'
    plt.clf()

    # 3D representation of all the segmented customers
    fig = plt.figure(1)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    ax.scatter(rfm_df_scaled['Frequency'], rfm_df_scaled['Recency'], rfm_df_scaled['Amount'],
               c=rfm['Cluster_Id'], s=200, cmap='spring', alpha=0.5, edgecolor='darkgrey')

    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Recency', fontsize=16)
    ax.set_zlabel('Monetary', fontsize=16)

    ax.set_title('3D Plot of RFM Segments', fontsize=20)

    plt.savefig('static/result9.png')
    datahtml += '<img src="../static/result9.png" alt="result9">'
    plt.clf()

    #########################
    f = open('templates/result.html', 'w')
    f.write(convertToHtml(datahtml))

    f.close()

    return render_template("result.html")


if __name__ == "__main__":
    # app.debug = True
    app.run()
