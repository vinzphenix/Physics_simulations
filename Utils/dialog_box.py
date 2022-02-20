#!/usr/bin/python

"""
Author: Kyle Kowalczyk
Description: Username and password entry box
Python Interpreter: v2.7.13
"""

from tkinter import *


def ask(mode, parameters):
    def clear_widget(_):  # arg should be called "event"
        # will clear out any entry boxes defined below when the user shifts
        # focus to the widgets defined below
        for box, name in zip(boxList, names):
            if box == main.focus_get() and box.get() == name:
                box.delete(0, END)

    def repopulate_defaults(_):
        # will repopulate the default text previously inside the entry boxes defined below if
        # the user does not put anything in while focused and changes focus to another widget
        for box, name in zip(boxList, names):
            if box != main.focus_get() and box.get() == '':
                box.insert(0, name)

    def login():  # arg should be called "*event"
        # Able to be called from a key binding or a button click because of the '*event'
        for box in boxList:
            parameters.append(box.get())
        main.quit()
        main.destroy()
        # If I wanted I could also pass the username and password I got above to another
        # function from here.

    # creates the main window object, defines its name, and default size
    main = Tk()
    main.title('Parameters of Simulation')
    if mode == 'animation':
        main.geometry('225x150')
    else:
        main.geometry('225x250')

    # defines a grid 50 x 50 cells in the main window
    rows = 0
    while rows < 10:
        main.rowconfigure(rows, weight=1)
        main.columnconfigure(rows, weight=1)
        rows += 1

    boxList = []
    names = ['Tend', 'l1', 'm1']
    if mode == 'draw':
        names += ['lw', 'cmap']
    for i, this_name in enumerate(names):
        this_box = Entry(main)
        this_box.insert(0, this_name)
        this_box.bind("<FocusIn>", clear_widget)
        this_box.bind('<FocusOut>', repopulate_defaults)
        this_box.grid(row=i, column=5, sticky='NS', pady=(5, 5))
        boxList.append(this_box)

    boxList[-1].bind('<Return>', login)

    # adds login button and defines its properties
    login_btn = Button(main, text='Login', command=login)
    login_btn.bind('<Return>', login)
    login_btn.grid(row=5, column=5, sticky='NESW')

    main.mainloop()
    return parameters


def __main__():
    """ Execute this file here """


if __name__ == "__main__":
    # Actually run your code in here
    print(ask('animation', []))
