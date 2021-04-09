from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from geopy.distance import geodesic


app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['SECRET_KEY'] = 'hard to guess string'

class NameForm(FlaskForm):
    Alat = StringField('What is A\'s latitude?', validators=[DataRequired()])
    Along = StringField('What is A\'s longtitude?', validators=[DataRequired()])
    Blat = StringField('What is B\'s latitude?', validators=[DataRequired()])
    Blong = StringField('What is B\'s longtitude?', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():
    dis = None
    form = NameForm()
    if form.validate_on_submit():
        Alat = form.Alat.data
        Along = form.Along.data
        Blat = form.Blat.data
        Blong = form.Blong.data

        A = Alat, Along
        B = Blat, Blong

        dis = geodesic(A, B).km


        form.Alat.data = ''
        form.Along.data = ''
        form.Blat.data = ''
        form.Blong.data = ''
    return render_template('main.html', form=form, dis = dis)


if __name__ == '__main__':
    app.run(debug=True)