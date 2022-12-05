
import bucket

path = './models/image-embedding/checkpoint/history.yaml'
history = bucket.loadYaml(path)

import plotly.graph_objects
import numpy
import os

class Chart:

    def __init__(self, style):

        self.style = style
        return

    def createFigure(self):

        self.figure = plotly.graph_objects.Figure()

    def addLine(self, track, name=None):

        length = len(track)
        line = plotly.graph_objects.Scatter(
            x=[i for i in range(length)], 
            y=track,
            name=name,
            mode='lines+markers'
        )
        self.figure.add_trace(line)
        return

    def showFigure(self):

        self.figure.show()
        return

    def saveFigure(self, path, form='html'):

        if(form=='html'):

            folder = os.path.dirname(path)
            os.makedirs(folder, exist_ok=True)
            self.figure.write_html(path)
            pass

        if(form=='png'):

            folder = os.path.dirname(path)
            os.makedirs(folder, exist_ok=True)
            self.figure.write_image(path)
            pass

        return

    pass

folder = './interim'
os.makedirs(folder, exist_ok=True)

chart = Chart(style='line')
chart.createFigure()
chart.addLine(history['train']['cost']['iteration']['loss'], 'train loss')
chart.addLine(history['validation']['cost']['iteration']['loss'], 'validation loss')
chart.addLine(history['test']['cost']['iteration']['loss'], 'test loss')
chart.saveFigure('./{}/loss.html'.format(folder), form='html')
chart.saveFigure('./{}/loss.png'.format(folder), form='png')

chart = Chart(style='line')
chart.createFigure()
chart.addLine(history['train']['cost']['iteration']['divergence'], 'train divergence')
chart.addLine(history['validation']['cost']['iteration']['divergence'], 'validation divergence')
chart.addLine(history['test']['cost']['iteration']['divergence'], 'test divergence')
chart.saveFigure('./{}/divergence.html'.format(folder), form='html')
chart.saveFigure('./{}/divergence.png'.format(folder), form='png')

chart = Chart(style='line')
chart.createFigure()
chart.addLine(history['train']['cost']['iteration']['reconstruction'], 'train reconstruction')
chart.addLine(history['validation']['cost']['iteration']['reconstruction'], 'validation reconstruction')
chart.addLine(history['test']['cost']['iteration']['reconstruction'], 'test reconstruction')
chart.saveFigure('./{}/reconstruction.html'.format(folder), form='html')
chart.saveFigure('./{}/reconstruction.png'.format(folder), form='png')

chart = Chart(style='line')
chart.createFigure()
chart.addLine(history['train']['cost']['iteration']['projection'], 'train projection')
chart.addLine(history['validation']['cost']['iteration']['projection'], 'validation projection')
chart.addLine(history['test']['cost']['iteration']['projection'], 'test projection')
chart.saveFigure('./projection.html', form='html')
chart.saveFigure('./projection.png', form='png')


# tr = bucket.loadPickle(path='./resource/ACNE04/Feedback/V2/train.pkl')
# tr.keys()
# tr['encoding'].shape



