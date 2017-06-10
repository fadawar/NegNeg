import jinja2
from flask import Flask, render_template, request, jsonify
from web.negneg import find_negation


app = Flask(__name__)
app.jinja_loader = jinja2.FileSystemLoader('web/templates')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('home.html',
                               results=find_negation(request.form['content']),
                               original=request.form['content'])
    return render_template('home.html')


@app.route('/json', methods=['POST'])
def negation():
    return jsonify(find_negation(request.form['content']))


if __name__ == '__main__':
    app.run(debug=True)
