from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

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
    cluster_number = 3

    # validation if has request
    # if request.method == 'POST':
    #     if request.form['cluster_number'] != '':
    #         cluster_number = int(request.form['cluster_number'])

    datahtml = ""
    
    # Load dataset
    df = pd.read_csv('data.csv', delimiter=';', skiprows=0, low_memory=False)
    # menampilkan sample data sebanyak 5 baris
    datahtml += "<h1>Sample Data</h1>" + df.head().to_html()

    # scatter plot
    plt.scatter(df['jumlah_transaksi'], df['total_penjualan'])
    plt.xlabel('Jumlah Transaksi')
    plt.ylabel('Total Penjualan')
    plt.savefig('static/scatter.png')
    datahtml += '<hr><p>Visualisasi dengan Scatter Plot</p><img src="../static/scatter.png" alt="scatter" width="500" height="500">'

    # k-means clustering
    km = KMeans(n_clusters=cluster_number)

    # melakukan prediksi
    y_predicted = km.fit_predict(df[['jumlah_transaksi','total_penjualan']])

    # hasil prediksi
    df['cluster'] = y_predicted
    datahtml += df.head().to_html()

    # 
    df1 = df[df.cluster == 0]
    df2 = df[df.cluster == 1]
    df3 = df[df.cluster == 2]

    plt.scatter(df1.jumlah_transaksi, df1.total_penjualan, color='green', label='cluster 0')
    plt.scatter(df2.jumlah_transaksi, df2.total_penjualan, color='red', label='cluster 1')
    plt.scatter(df3.jumlah_transaksi, df3.total_penjualan, color='blue', label='cluster 2')
    plt.xlabel('Jumlah Transaksi')
    plt.ylabel('Total Penjualan')
    plt.legend()
    plt.savefig('static/scatter_cluster.png')
    plt.clf()
    datahtml += '<img src="../static/scatter_cluster.png" alt="scatter" width="500" height="500">'

    # menskalakan data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df[['total_penjualan']])
    df['total_penjualan_cluster']= scaler.transform(df[['total_penjualan']])

    scaler.fit(df[['jumlah_transaksi']])
    df['jumlah_transaksi_cluster'] = scaler.transform(df[['jumlah_transaksi']])

    # k-means clustering
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(df[['jumlah_transaksi','total_penjualan']])

    
    # menampilkan hasil prediksi
    df1 = df[df.cluster == 0]
    df2 = df[df.cluster == 1]
    df3 = df[df.cluster == 2]

    plt.scatter(df1.jumlah_transaksi, df1.total_penjualan, color='green', label='cluster 0')
    plt.scatter(df2.jumlah_transaksi, df2.total_penjualan, color='red', label='cluster 1')
    plt.scatter(df3.jumlah_transaksi, df3.total_penjualan, color='blue', label='cluster 2')
    plt.xlabel('Jumlah Transaksi')
    plt.ylabel('Total Penjualan')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*',label='centroid' )
    plt.legend()
    plt.savefig('static/scatter_cluster2.png')
    plt.clf()
    datahtml += '<img src="../static/scatter_cluster2.png" alt="scatter" width="500" height="500">'

    # k range 1-10
    k_rng = range(1,10)
    sse = []

    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(df[['jumlah_transaksi','total_penjualan']])
        sse.append(km.inertia_)
    
    plt.xlabel('K')
    plt.ylabel('Sum of Squared Error')
    plt.plot(k_rng, sse)
    plt.savefig('static/elbow.png')
    datahtml += '<img src="../static/elbow.png" alt="scatter" width="500" height="500">'

    selected_cols = ["jumlah_transaksi","total_penjualan"]
    cluster_data = df.loc[:,selected_cols]

    kmeans_sel = KMeans(init='k-means++', n_clusters=3, n_init=100, random_state=2).fit(cluster_data)
    labels = pd.DataFrame(kmeans_sel.labels_)
    clustered_data = cluster_data.assign(Cluster=labels)

    grouped_km = clustered_data.groupby(['Cluster']).mean().round(1)
    datahtml += grouped_km.to_html()

    datahtml += "<h1>Hasil Clustering</h1>"
    # limit 5 data
    datahtml += "<p>Cluster 0</p>"
    datahtml += df[df.cluster == 0].head().to_html()
    datahtml += "<p>Cluster 1</p>"
    datahtml += df[df.cluster == 1].to_html()
    datahtml += "<p>Cluster 2</p>"
    datahtml += df[df.cluster == 2].to_html()

    # sort by cluster
    # df.sort_values(by=['cluster'])
    # datahtml += df.to_html()
    


    f = open('templates/result.html', 'w')
    f.write(convertToHtml(datahtml))

    f.close()

    return render_template("result.html")


if __name__ == "__main__":
    app.debug = True
    app.run()
