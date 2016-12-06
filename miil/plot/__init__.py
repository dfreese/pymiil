import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

class FixedOrderFormatter(ScalarFormatter):
    """
    Formats axis ticks using scientific notation with a constant order of
    magnitude

    Taken from:
    http://stackoverflow.com/a/3679918/2465202
    """
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """
        Over-riding this to avoid having orderOfMagnitude reset elsewhere
        """
        self.orderOfMagnitude = self._order_of_mag

def set_y_axis_order(order, visible=False):
    y_ax = plt.gca().yaxis
    y_ax.set_major_formatter(FixedOrderFormatter(order))
    # Set whether the order of magnitude offset is visible or not
    # taken from http://stackoverflow.com/a/38207800/2465202
    plt.setp(y_ax.get_offset_text(), visible=visible)

