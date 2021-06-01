import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.random.rand(10))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect('equal')

def onclick(event):
    circle=plt.Circle((event.xdata,event.ydata),0.05,color='black')
    ax.add_patch(circle)
    fig.canvas.draw()
    st.write(f'x = {event.xdata}, y = {event.ydata}')
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)
#plt.show()
st.write(fig)