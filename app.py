from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

from flask import Flask, render_template, request

app = Flask(__name__)


def convertToHtml(result,data):
    html = """
    <!DOCTYPE html>
    <html lang="en">

    <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-GLhlTQ8iRABdZLl6O3oVMWSktQOp6b7In1Zl3/Jr59b6EGGoI1aFkw7cmDA6j6gD" crossorigin="anonymous">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/v/bs5/dt-1.13.1/datatables.min.css"/>
    </head>

    <body>
    <ul class="nav nav-tabs" id="myTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="result-tab" data-bs-toggle="tab" data-bs-target="#result" type="button" role="tab" aria-controls="result" aria-selected="true">Result</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="details-tab" data-bs-toggle="tab" data-bs-target="#details" type="button" role="tab" aria-controls="details" aria-selected="false">Show Details</button>
        </li>
    </ul>
    <div class="tab-content" id="myTabContent">
        <div class="tab-pane fade show active" id="result" role="tabpanel" aria-labelledby="result-tab">
        """ + result + """
        </div>
        <div class="tab-pane fade" id="details" role="tabpanel" aria-labelledby="details-tab">
            """ + data + """
        </div>
    </div>
    

    <script src="https://code.jquery.com/jquery-3.6.3.min.js" integrity="sha256-pvPw+upLPUjgMXY0G+8O0xUf+/Im1MZjXxxgOcBQBXU=" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js" integrity="sha384-w76AqPfDkMBDXo30jS1Sgez6pr3x5MlQ1ZAGC+nuZB+EYdgRZgiwxhTBTkF7CXvN" crossorigin="anonymous"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/v/bs5/dt-1.13.1/datatables.min.js"></script>
    <script>
        window.onload = function(){
            document.getElementByTagName('table').classList.add('table')
        }
        $(document).ready(function () {
            $('#resulttbl').DataTable();
            $('#resulttbl2').DataTable();
            $('#resulttbl3').DataTable();
        });
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

    labelsArr = [
        "Terlaris",
        "Laris",
        "Kurang Laris"
    ]

    selectedLabel = int(request.form['pilihan_cluster'])

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
    plt.clf()
    datahtml += '<hr><p>Visualisasi dengan Scatter Plot</p><img src="../static/scatter.png" alt="scatter" width="500" height="500">'

    # k-means clustering
    km = KMeans(n_clusters=cluster_number, n_init='auto')

    # melakukan prediksi
    y_predicted = km.fit_predict(df[['jumlah_transaksi','total_penjualan']])

    # hasil prediksi
    df['cluster'] = y_predicted
    datahtml += df.head().to_html()

    # pemberian label cluster
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
    km = KMeans(n_clusters=cluster_number, n_init='auto')
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
        km = KMeans(n_clusters=k, n_init='auto')
        km.fit(df[['jumlah_transaksi','total_penjualan']])
        sse.append(km.inertia_)
    
    plt.xlabel('K')
    plt.ylabel('Sum of Squared Error')
    plt.plot(k_rng, sse)
    plt.savefig('static/elbow.png')
    plt.clf()
    datahtml += '<img src="../static/elbow.png" alt="scatter" width="500" height="500">'

    selected_cols = ["jumlah_transaksi","total_penjualan"]
    cluster_data = df.loc[:,selected_cols]

    kmeans_sel = KMeans(init='k-means++', n_clusters=cluster_number, n_init='auto', random_state=2).fit(cluster_data)
    labels = pd.DataFrame(kmeans_sel.labels_)
    clustered_data = cluster_data.assign(Cluster=labels)

    grouped_km = clustered_data.groupby(['Cluster']).mean().round(1)
    datahtml += grouped_km.to_html()

    datahtml += "<h1>Hasil Clustering</h1>"
    # limit 5 data
    datahtml += "<p>Cluster 0</p>"
    datahtml += df[df.cluster == 0].head().to_html()
    datahtml += "<p>Cluster 1</p>"
    datahtml += df[df.cluster == 1].head().to_html()
    datahtml += "<p>Cluster 2</p>"
    datahtml += df[df.cluster == 2].head().to_html()

    # labeling cluster
    maxjmltransaksicluster1 = df[df.cluster == 0].jumlah_transaksi_cluster.max()
    maxjmltransaksicluster2 = df[df.cluster == 1].jumlah_transaksi_cluster.max()
    maxjmltransaksicluster3 = df[df.cluster == 2].jumlah_transaksi_cluster.max()

    # new array
    labelarr = []
    newarr = [maxjmltransaksicluster1,maxjmltransaksicluster2,maxjmltransaksicluster3]
    oldarr = [maxjmltransaksicluster1,maxjmltransaksicluster2,maxjmltransaksicluster3]

    # sort array
    newarr.sort(reverse=True) # patokan label

    idx = 3
    if selectedLabel >= 0 and selectedLabel <= 2:
        for i in range(len(oldarr)):
            if oldarr[i] == newarr[selectedLabel]:
                idx = i

    if idx >= 0 and idx < len(newarr):
        result = "Hasil Clustering"   
        result += "<br>"
        kodeBrg = df[df.cluster == idx].kode_barang
        namaBrg = df[df.cluster == idx].nama_barang
        jmlTrans = df[df.cluster == idx].jumlah_transaksi
        totPenj = df[df.cluster == idx].total_penjualan
        result += "<table id='resulttbl' class='table table-striped table-hover'>"
        result += "<thead><tr>"
        result += "<th>Kode Barang</th>"
        result += "<th>Nama Barang</th>"
        result += "<th>Jumlah Transaksi</th>"
        result += "<th>Total Penjualan</th>"
        result += "</tr></thead><tbody>"
        for i in range(len(kodeBrg)):
            result += "<tr>"
            result += "<td>"+kodeBrg.iloc[i]+"</td>"
            result += "<td>"+str(namaBrg.iloc[i])+"</td>"
            result += "<td>"+str(jmlTrans.iloc[i])+"</td>"
            result += "<td>"+str(totPenj.iloc[i])+"</td>"
            result += "</tr>"
        result += "</tbody></table>"
    else:
        result = '<hr>'
        result += "Hasil Clustering 1"
        result += "<br>"
        kodeBrg = df[df.cluster == 0].kode_barang
        namaBrg = df[df.cluster == 0].nama_barang
        jmlTrans = df[df.cluster == 0].jumlah_transaksi
        totPenj = df[df.cluster == 0].total_penjualan
        result += "<table id='resulttbl' class='table table-striped table-hover'>"
        result += "<thead><tr>"
        result += "<th>Kode Barang</th>"
        result += "<th>Nama Barang</th>"
        result += "<th>Jumlah Transaksi</th>"
        result += "<th>Total Penjualan</th>"
        result += "</tr></thead><tbody>"
        for i in range(len(kodeBrg)):
            result += "<tr>"
            result += "<td>"+kodeBrg.iloc[i]+"</td>"
            result += "<td>"+str(namaBrg.iloc[i])+"</td>"
            result += "<td>"+str(jmlTrans.iloc[i])+"</td>"
            result += "<td>"+str(totPenj.iloc[i])+"</td>"
            result += "</tr>"
        result += "</tbody></table>"
        result += "Hasil Clustering 2"
        result += "<br>"
        kodeBrg = df[df.cluster == 1].kode_barang
        namaBrg = df[df.cluster == 1].nama_barang
        jmlTrans = df[df.cluster == 1].jumlah_transaksi
        totPenj = df[df.cluster == 1].total_penjualan
        result += "<table id='resulttbl2' class='table table-striped table-hover'>"
        result += "<thead><tr>"
        result += "<th>Kode Barang</th>"
        result += "<th>Nama Barang</th>"
        result += "<th>Jumlah Transaksi</th>"
        result += "<th>Total Penjualan</th>"
        result += "</tr></thead><tbody>"
        for i in range(len(kodeBrg)):
            result += "<tr>"
            result += "<td>"+kodeBrg.iloc[i]+"</td>"
            result += "<td>"+str(namaBrg.iloc[i])+"</td>"
            result += "<td>"+str(jmlTrans.iloc[i])+"</td>"
            result += "<td>"+str(totPenj.iloc[i])+"</td>"
            result += "</tr>"
        result += "</tbody></table>"
        result += "Hasil Clustering 3"
        result += "<br>"
        kodeBrg = df[df.cluster == 2].kode_barang
        namaBrg = df[df.cluster == 2].nama_barang
        jmlTrans = df[df.cluster == 2].jumlah_transaksi
        totPenj = df[df.cluster == 2].total_penjualan
        result += "<table id='resulttbl3' class='table table-striped table-hover'>"
        result += "<thead><tr>"
        result += "<th>Kode Barang</th>"
        result += "<th>Nama Barang</th>"
        result += "<th>Jumlah Transaksi</th>"
        result += "<th>Total Penjualan</th>"
        result += "</tr></thead><tbody>"
        for i in range(len(kodeBrg)):
            result += "<tr>"
            result += "<td>"+kodeBrg.iloc[i]+"</td>"
            result += "<td>"+str(namaBrg.iloc[i])+"</td>"
            result += "<td>"+str(jmlTrans.iloc[i])+"</td>"
            result += "<td>"+str(totPenj.iloc[i])+"</td>"
            result += "</tr>"
        result += "</tbody></table>"



    # result += df[df.cluster == 0].head().to_html()


    f = open('templates/result.html', 'w')
    f.write(convertToHtml(result,datahtml))

    f.close()

    return render_template("result.html")


if __name__ == "__main__":
    app.debug = True
    app.run()
