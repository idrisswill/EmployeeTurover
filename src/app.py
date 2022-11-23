from io import BytesIO
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect, Response, flash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_bootstrap import Bootstrap5
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
import os
import json

import fucntion.functions as funcs

app = Flask(__name__)
app.config['SECRET_KEY'] = 'la clef de lenfer'
app.config['UPLOAD_DIRECTORY'] = 'uploads/'
app.config['DATASET_DIR'] = '/data/AllProjectIA/EmployeesTurnover/src/uploads'
app.config['ALLOWED_EXTENSIONS'] = ['.csv', '.tsv', '.xml', '.xlsx', '.xltx', '.xls', '.xls']
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
boostrap = Bootstrap5(app)


@app.route("/")
@app.route("/home")
def index():
    datasets = funcs.dataset_description(app.config['DATASET_DIR'])
    return render_template("index.html", datasets=datasets)


@app.route("/about")
def about():
    return render_template("about.html", title='About')


@app.route("/upload", methods=['POST'])
def upload_data():
    result = request.args.get('result')
    if result == "True":
        flash(' le fichier a ete bien ajouté', 'success')
    else:
        flash(' une erreur veillez essayer', 'danger')
    try:
        file = request.files['file']
        extension = os.path.splitext(file.filename)[1].lower()
        if file:
            if extension not in app.config['ALLOWED_EXTENSIONS']:
                flash('File is not an data format.', 'danger')
                return redirect('/')
            file.save(os.path.join(
                app.config['UPLOAD_DIRECTORY'],
                secure_filename(file.filename)
            ))
    except RequestEntityTooLarge:
        flash('File is large than the 200 MB', 'danger')
        return redirect('/')
    return redirect('/')


@app.route("/distribution/<dataset_use>", defaults={'select_target': None, 'select_data': None})
@app.route("/distribution/<dataset_use>/<select_target>", defaults={'select_data': None})
@app.route("/distribution/<dataset_use>/<select_target>/<select_data>")
@app.route("/distribution/<dataset_use>/<select_target>/<select_data>/<data_scale>")
def estimate_distribution(dataset_use, select_target, select_data, data_scale=None):
    datasets = funcs.dataset_description(app.config['DATASET_DIR'])
    data = pd.read_csv(f"{app.config['DATASET_DIR']}/{dataset_use}.csv")
    columns = data.columns
    columns_data = []
    if select_target is not None:
        columns_data = columns.to_list()
        columns_data.remove(select_target)

    return render_template("about.html",
                           datasets=datasets,
                           columns=columns,
                           select_data=select_data,
                           columns_data=columns_data,
                           dataset_usage=dataset_use,
                           select_target=select_target,
                           scale=data_scale
                           )


@app.route("/select_dataset", methods=['POST'])
def select_dataset():
    select = request.form.get('dataset_select')
    return redirect(url_for('estimate_distribution', dataset_use=(str(select))))


@app.route("/select_target/<dataset_use>", methods=['POST'])
def select_target(dataset_use):
    select = request.form.get('select_target')
    return redirect(url_for('estimate_distribution',
                            dataset_use=dataset_use,
                            select_target=(str(select))
                            ))


@app.route("/select_data_column/<dataset_use>/<select_target>", methods=['POST'])
def select_data(dataset_use, select_target):
    select = request.form.get('select_data')
    print(f'la valeur est: {str(select)}')
    return redirect(url_for('estimate_distribution',
                            dataset_use=dataset_use,
                            select_target=select_target,
                            select_data=(str(select))
                            ))


@app.route("/scale_data/<dataset_use>/<select_target>/<select_data>", methods=['POST'])
def data_scale(dataset_use, select_target, select_data):
    select = request.form.get('scale')
    print(f'la valeur est: {str(select)}')
    return redirect(url_for('estimate_distribution',
                            dataset_use=dataset_use,
                            select_target=select_target,
                            select_data=select_data,
                            scale=(str(select))
                            ))


@app.route('/chart')
def chart():
    param = json.loads(request.args.get("param"))
    print(param)
    fig = plt.figure(constrained_layout=True)
    gspec = fig.add_gridspec(ncols=3, nrows=2)
    ax = fig.add_subplot(gspec[0, :])

    data = pd.read_csv(f"{app.config['DATASET_DIR']}/{param['dataset']}.csv")
    datatOgIVE = data[data[param['select_target']] == True][param['select_data']]

    datatOgIVE.plot(ax=ax, kind='hist', bins=50, density=True, alpha=0.5,
                         color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])
    dataYLim = ax.get_ylim()
    best_distibutions = funcs.best_fit_distribution(datatOgIVE, 200, ax)
    best_dist = best_distibutions[0]
    ax.set_ylim(dataYLim)
    ax.set_title(u'tracer de la courbe.\n All Fitted Distributions')
    ax.set_xlabel(u'numvotes')
    ax.set_ylabel('Frequency')

    pdf = funcs.make_pdf(dist=best_dist[0], params=list(best_dist[1]), size=3680)
    ax1 = fig.add_subplot(gspec[1, :])
    pdf.plot(lw=2, label='PDF', legend=True, ax=ax1)
    datatOgIVE.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax1)

    param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_dist[1])])
    dist_str = '{}({})'.format(best_dist[0].name, param_str)
    ax1.set_title(u' best fit distribution \n' + dist_str)
    ax1.set_xlabel(u'numvote')
    ax1.set_ylabel('Frequency')
    output = BytesIO()
    Canvas(fig).print_png(output)

    return Response(output.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
