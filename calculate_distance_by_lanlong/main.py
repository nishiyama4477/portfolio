from flask import Flask, render_template, session, redirect, url_for, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_moment import Moment
from datetime import datetime
from wtforms import FloatField,SubmitField
from wtforms.validators import DataRequired, NumberRange
from geopy.distance import geodesic

app = Flask(__name__)
bootstrap = Bootstrap(app)
moment  = Moment(app)
app.config['SECRET_KEY'] = 'hard to guess string'

class NameForm(FlaskForm):
    Alat = FloatField('What is A\'s latitude?', validators=[DataRequired(), NumberRange(min=-90, max=90, message='Latitude is in between -90 to 90')])
    Along = FloatField('What is A\'s longtitude?', validators=[DataRequired(), NumberRange(min=-180, max=180, message='Longitude is in between -180 to 180')])
    Blat = FloatField('What is B\'s latitude?', validators=[DataRequired(), NumberRange(min=-90, max=90, message='Latitude is in between -90 to 90')])
    Blong = FloatField('What is B\'s longtitude?', validators=[DataRequired(), NumberRange(min=-180, max=180, message='Longitude is in between -180 to 180')])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = NameForm()
    if form.validate_on_submit():
        Alat = form.Alat.data
        Along = form.Along.data
        Blat = form.Blat.data
        Blong = form.Blong.data
        if Alat == Along == Blat == Blong and Alat != 0 and Along != 0 and Blat != 0 and Blong != 0:
            session['A'] = Alat, Along
            session['B'] = Blat, Blong
            session['dis'] = str(0)
        else:
            session['A'] = Alat, Along
            session['B'] = Blat, Blong
            session['dis'] = geodesic(session['A'], session['B']).km

        # old_dis = session.get('dis')
        # if old_dis is not None and old_dis != geodesic(A, B).km:
        #     flash('Looks like you have changed your location!')
        # session['dis'] = geodesic(A,B).km

        return redirect(url_for('index'))
    return render_template('main.html', form=form, dis=session.get('dis'), current_time=datetime.utcnow())


if __name__ == '__main__':
    app.run(debug=True)