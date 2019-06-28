from flask import Flask, render_template, redirect, url_for, request
from Collaborated import *

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['POST', 'GET'])
def main():
    return render_template('main.html')

@app.route('/newpage', methods=['POST','GET'])
def newpage():
    text = request.form.get('inputText')
    date = request.form.get('datepicker')
    fdata = initiate(text,date)
    return render_template('newpage.html', data = fdata)


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATE_AUTO_RELOAD'] = True
    app.run(debug=True)
