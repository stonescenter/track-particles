#function to plot tracks
def track_plot_xyz(list_of_df = [], **kwargs):
    
    global pivot, shift
    
    n_tracks = 1
    title = 'Track plots'
    path = 'chart.html'
    opacity = 0.5
    marker_size = 3
    line_size = 3
    len_list_df = len(list_of_df)
    list_of_colors = ['red','blue', 'green', 'magenta', 'chocolate',
                      'teal', 'indianred', 'yellow', 'orange', 'silver']
    
    assert (len_list_df <= len(list_of_colors)), 'The list must contain less than 10 dataframes.'
    
    if kwargs.get('n_tracks'):
        n_tracks = kwargs.get('n_tracks')
        if n_tracks > list_of_df[0].shape[0]:
            n_tracks = abs(list_of_df[0].shape[0])
            wrn_msg = ('The number of tracks to plot is greater than the number of tracks in '
                       'the dataframe.\nn_tracks will be: ' +  str(n_tracks) + 
                       ' (the number of tracks in the dataset)')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)
                
    if kwargs.get('pivot'):
        pivot = kwargs.get('pivot')
    
    if kwargs.get('title'):
        title = kwargs.get('title')
        
    if kwargs.get('opacity'):
        opacity = kwargs.get('opacity')
        if opacity > 1.0:
            opacity = 1.0
            wrn_msg = ('The opacity value is greater than 1.0\n'
                       'The opacity value is will be set with 1.0 value.')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)
    
    if kwargs.get('marker_size'):
        marker_size = abs(kwargs.get('marker_size'))
        
    if kwargs.get('line_size'):
        line_size = abs(kwargs.get('line_size'))
        
    
    
    len_xyz = 5
    

    # Initializing lists of indexes
    selected_columns_x = np.zeros(len_xyz)
    selected_columns_y = np.zeros(len_xyz)
    selected_columns_z = np.zeros(len_xyz)

    # Generating indexes
    for i in range(len_xyz):
        selected_columns_x[i] = int(i * 3 + 0)
        selected_columns_y[i] = int(i * 3 + 1)
        selected_columns_z[i] = int(i * 3 + 2)

    # list of data to plot
    data = []
    track = [None] * n_tracks
    
    for i in range(len_list_df):
        try:
            df_name = str(list_of_df[i].name)
        except:
            df_name = 'track[' + str(i) + ']'
            warnings.warn('For a better visualization, set the name of dataframe to plot:'
                          '\nE.g.: df.name = \'track original\'', 
                          RuntimeWarning, stacklevel=2)
        
        for j in range(n_tracks):
            track[j] = go.Scatter3d(
                # Removing null values (zeroes) in the plot
                x = list_of_df[i].replace(0.0, np.nan).iloc[j, selected_columns_x],
                y = list_of_df[i].replace(0.0, np.nan).iloc[j, selected_columns_y],
                z = list_of_df[i].replace(0.0, np.nan).iloc[j, selected_columns_z],
                name = df_name + ' ' + str(j),
                opacity = opacity,
                marker = dict(
                    size = marker_size,
                    opacity = opacity,
                    color = list_of_colors[i],
                ),
                line = dict(
                    color = list_of_colors[i],
                    width = line_size
                )
            )
            # append the track[i] in the list for plotting
            data.append(track[j])
    layout = dict(
        #width    = 900,
        #height   = 750,
        autosize = True,
        title    = title,
        scene = dict(
            xaxis = dict(
                gridcolor       = 'rgb(255, 255, 255)',
                zerolinecolor   = 'rgb(255, 255, 255)',
                showbackground  = True,
                backgroundcolor = 'rgb(230, 230,230)',
                title           ='x (mm)'
            ),
            yaxis=dict(
                gridcolor       = 'rgb(255, 255, 255)',
                zerolinecolor   = 'rgb(255, 255, 255)',
                showbackground  = True,
                backgroundcolor = 'rgb(230, 230,230)',
                title           = 'y (mm)'
            ),
            zaxis=dict(
                gridcolor       = 'rgb(255, 255, 255)',
                zerolinecolor   = 'rgb(255, 255, 255)',
                showbackground  = True,
                backgroundcolor = 'rgb(230, 230,230)',
                title           = 'z (mm)'
            ),             
            camera = dict(
                up = dict(
                    x = 1,
                    y = 1,
                    z = -0.5
                ),
                eye = dict(
                    x = -1.7428,
                    y = 1.0707,
                    z = 0.7100,
                )
            ),
            aspectratio = dict( x = 1, y = 1, z = 1),
            aspectmode = 'manual'
            
        ),   
    )

    fig =  go.Figure(data = data, layout = layout)
    #init_notebook_mode(connected=False)
    
    if kwargs.get('path'):
        path = kwargs.get('path')
        fig.write_html(path) 

    return fig   