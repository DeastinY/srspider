#!/usr/bin/python3
"""
sample of creating a radar chart (a.k.a. a spider or star chart) [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from random import randint

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def interactive_data():
    print('Name:')
    name = input()
    #criteria = ['Athletik','Beschwören','Biotech','Cracken','Einfluss','Elektronik','Feuerwaffen','Heimlichkeit','Hexerei','Mechanik','Nahkampf','Natur','Schauspielerei','Tasken','Verzaubern','Riggen','Speziell']
    criteria = ['Infiltration', 'Nahkampf', 'Fernkampf', 'Face', 'Rigger', 'Hacker', 'Zauberer', 'Adept']
    data = []
    for c in criteria :
        d = None
        while d not in ['0','1','2','3','4','5']:
            print('Enter rating for {}'.format(c))
            d = input()
        data.append(int(d))
    l = lambda criteria, data: [print("{}: {}\t{}".format(i, d[1], d[0])) for i,d in enumerate(zip(criteria, data))]
    i = None
    print('Enter the number to edit, l to list or q to quit and show the result !')
    l(criteria, data)
    while i is None or i != 'q':
        i = input()
        if i == 'q' or i == 'Q':
            break
        elif i == 'l' or i == 'L':
            l(criteria, data)
        elif len(i)>0 and i.isdigit() and int(i) < len(criteria):
            i = int(i)
            while True:
                print('Enter new value for {}'.format(criteria[i]))
                temp = input()
                if len(temp) == 0 or not temp.isdigit():
                    print('Bad input')
                else:
                    temp=int(temp)
                    if temp < 6 and temp >= 0:
                        data[i] = temp
                        print('Updated to {}'.format(data[i]))
                        break

    return [criteria, (name, [data])]

def example_coffee():
    return [
        ['Dry Fragrance', 'Brightness/Acidity', 'Flavor', 'Body/Mouthfeel', 'Sweetness', 'Complexity', 'Price-Performance'],
            ('Example Coffee', [[randint(0,5) for i in range(7)]])
        ]

if __name__ == '__main__':
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)

    data = interactive_data()
    N = len(data[0])
    theta = radar_factory(N)

    spoke_labels = data.pop(0)
    spoke_labels = [s.replace('/','\n') for s in spoke_labels]

    fig = plt.figure(figsize=(7, 7))

    colors = ['b', 'r', 'g', 'm', 'y']
    title, case_data = data[0]
    case_data[0] = [c+1 for c in case_data[0]]
    ax = fig.add_subplot(1, 1, 1, projection='radar')
    ax.set_ylim(bottom=0, top=5)
    # labels = ('1 Really Bad', '2 Bad', '3 Neutral', '4 Good', '5 Really Good')
    # plt.rgrids([1, 2, 3, 4, 5])
    plt.rgrids([1,2,3,4,5,6], [])
    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    plt.figtext(0.5, 0.965, title,
                ha='center', color='black', weight='bold', size='large')
    #plt.figtext(0.5, 0.05, 'Average {:.2f}'.format(sum(case_data[0])/len(case_data[0])),
    #            ha='center', color='black' )
    plt.show()
