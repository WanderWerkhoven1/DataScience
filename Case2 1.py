import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Dataframe laden
df = pd.read_csv('FundaEndDataFrame.csv')

# Titel van de webapp
st.title('Data Analyse van Huizenprijzen')

# maken van sidebar
st.sidebar.title('In welke waarde ben je geïnteresseerd?')
option = st.sidebar.radio('Maak een keuze', ['Home pagina', 'Energielabel', 'Vierkante meter prijzen', "Regio's"])

# geselecteerde pagina opstellen voor Homepage
if option == 'Home pagina':
    # Pagina-titel en introductie
    st.title("Welkom op de Thuispagina")

    st.write("Wat leuk dat je op onze webapp bent beland. In deze webapp heb je de mogelijkheid om meerdere visualisaties te maken over de huizenmarkt in Amsterdam! ")
    st.write("Door een keuze te maken in het menu hiernaast kunt je kijken naar verschillende gebieden van interesse. ")
    st.write("Wil je weten hoe het zit met de Energielabels van huizen in Amsterdam? Klik dan op 'Energielabels'.")
    st.write("Ben je benieuwd naar de vierkante meter prijzen voor huizen in Amsterdam? Klik dan op 'Vierkante meter prijzen'.")
    st.write("Wil jij weten wat er gebeurt met de huizen in bepalde regios van Amsterdam? Klik dan op 'Regios'.")

# geselecteerde pagina opstellen voor Regio #####################################
elif option == 'regios':
    st.title("Regios")
    tabs = st.tabs(['Taartdiagram', "Staafdiagram", "Boxplot"])

    # Tab voor 'Taart diagram'
    with tabs[0]:
        st.write("Hier kun je de taart diagram zien van de hoeveelheid huizen per regio.")

        # Piechart dataframe op regio
        aantal_huizen_per_regio = df.groupby('regio')['prijzen'].count()
        verschillende_regio = ['Amsterdam-Noord', 'Amsterdam-West', 'Amsterdam-Zuid',
                               'Amsterdam Nieuw-West', 'Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)',
                               'Amsterdam-Oost', 'Amsterdam-Zuid-Oost', 'Oostelijk Havengebied (Amsterdam-Oost)',
                               'Amsterdam Westpoort']
        fig = px.pie(values=aantal_huizen_per_regio, names=verschillende_regio,
                     title='Percentage huizen te koop per regio')
        # Show the plot
        st.plotly_chart(fig)

    # Tab voor Staafdiagram
    with tabs[1]:
        # Staafdiagram pagina tekst
        st.write(
            'hieronder is een interactief staafdiagram weergegeven die de regios en de gemiddelde huizenprijs per regio aangeeft. Klik er lekker doorheen!')

        # Create the basic figure
        fig = go.Figure()
        # Loop through the different regions
        for regio in ['Amsterdam-Noord', 'Amsterdam-West', 'Amsterdam-Zuid',
                      'Amsterdam Nieuw-West', 'Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)',
                      'Amsterdam-Oost', 'Amsterdam-Zuid-Oost', 'Oostelijk Havengebied (Amsterdam-Oost)',
                      'Amsterdam Westpoort']:
            # Subset the DataFrame
            df_regio = df[df.regio == regio]
            # Add a trace for each regio subset
            fig.add_trace(go.Bar(
                x=df_regio['regio'],
                y=df_regio.groupby('regio')['prijzen'].mean(),
                name=regio))
        dropdown_buttons = [
            {'label': "Alle regio's", 'method': "update", 'args': [
                {"visible": [True, True, True, True, True, True, True, True, True]}, {"title": "Alle regio's"}]},
            {'label': "Amsterdam-Noord", 'method': "update", 'args': [
                {"visible": [True, False, False, False, False, False, False, False, False]}, {"title": "Amsterdam-Noord"}]},
            {'label': "Amsterdam-West", 'method': "update", 'args': [
                {"visible": [False, True, False, False, False, False, False, False, False]}, {"title": "Amsterdam-West"}]},
            {'label': "Amsterdam-Zuid", 'method': "update", 'args': [
                {"visible": [False, False, True, False, False, False, False, False, False]}, {"title": "Amsterdam-Zuid"}]},
            {'label': "Amsterdam Nieuw-West", 'method': "update", 'args': [
                {"visible": [False, False, False, True, False, False, False, False, False]},
                {"title": "Amsterdam Nieuw-West"}]},
            {'label': "Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)", 'method': "update", 'args': [
                {"visible": [False, False, False, False, True, False, False, False, False]},
                {"title": "Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)"}]},
            {'label': "Amsterdam-Oost", 'method': "update", 'args': [
                {"visible": [False, False, False, False, False, True, False, False, False]}, {"title": "Amsterdam-Oost"}]},
            {'label': "Amsterdam-Zuid-Oost", 'method': "update", 'args': [
                {"visible": [False, False, False, False, False, False, True, False, False]},
                {"title": "Amsterdam-Zuid-Oost"}]},
            {'label': "Oostelijk Havengebied (Amsterdam-Oost)", 'method': "update", 'args': [
                {"visible": [False, False, False, False, False, False, False, True, False]},
                {"title": "Oostelijk Havengebied (Amsterdam-Oost)"}]},
            {'label': "Amsterdam Westpoort", 'method': "update", 'args': [
                {"visible": [False, False, False, False, False, False, False, False, True]},
                {"title": "Amsterdam Westpoort"}]}
        ]
        # Update the figure to add dropdown menu
        fig.update_layout({
            'updatemenus': [{
                'type': 'dropdown',
                'x': 1, 'y': 1.1,
                'showactive': True, 'active': 0,
                'buttons': dropdown_buttons
            }],
            'width': 1000,  # Aanpassen naar gewenste breedte
            'height': 600,  # Aanpassen naar gewenste hoogte
            'xaxis_title': 'Regio',  # X-as label
            'yaxis_title': 'Gemiddelde Huizenprijzen',  # Y-as label
            'title': 'Gemiddelde Huizenprijzen per Regio'  # Titel van de plot
        })
        # Show the plot
        st.plotly_chart(fig)

    # Tab voor Boxplot
    with tabs[2]:
        st.write('Je bevindt je bij de Boxplot van de huizenprijzen voor verschillende regios')

        # Boxplot van huizenprijzen per regio
        fig = px.box(df, x='regio', y='prijzen', color='regio', title='Boxplot van huizenprijzen per regio')
        fig.update_layout(xaxis_title='Regio', yaxis_title='Huizenprijzen')
        st.plotly_chart(fig)

# Streamlit-applicatie
elif option == 'Vierkante meter prijzen':
    st.title('Scatterplot prijs en vierkante meter per regio')
    tabs2 = st.tabs(['Scatter zoeken', 'Scatter overview'])

# Tab voor 'Vierkante meter prijzen'
    with tabs2[0]:

    # Staafdiagram pagina tekst
        st.write(
        'Hieronder is een interactief staafdiagram weergegeven die de regios en de gemiddelde huizenprijs per regio aangeeft. Klik er lekker doorheen!')

    # Sliders voor Vierkante Meters en Prijzen
        min_vm, max_vm = st.slider('Vierkante meters',
                               min_value=0,  # Startwaarde aangepast naar 0
                               max_value=int(max(df['vierkante_meter'])),
                               value=(int(min(df['vierkante_meter'])), int(max(df['vierkante_meter']))),
                               step=25)  # Sliden per 25 m²

        min_prijs, max_prijs = st.slider('Prijs',
                                     min_value=200000,  # Startwaarde aangepast naar 200000
                                     max_value=int(max(df['prijzen'])),
                                     value=(int(min(df['prijzen'])), int(max(df['prijzen']))),
                                     step=25000)  # Sliden per 50.000

        # Filter de gegevens op basis van geselecteerde sliders
        filtered_data = df[
        (df['vierkante_meter'] >= min_vm) & (df['vierkante_meter'] <= max_vm) &
        (df['prijzen'] >= min_prijs) & (df['prijzen'] <= max_prijs)
         ]

        # Scatterplot maken met Plotly Express
        fig = px.scatter(filtered_data, x='vierkante_meter', y='prijzen', color='regio',
                     title='Scatterplot van Vierkante Meters vs Prijzen',
                     labels={'vierkante_meter': 'Vierkante Meters', 'prijzen': 'Prijzen', 'regio': 'Regio'})

        # Plot weergeven met Streamlit
        st.plotly_chart(fig)

        # Tab voor 'Vierkante meter prijzen'
        with tabs2[1]:

    # Staafdiagram pagina tekst
            st.write(
        'Hieronder is een interactief staafdiagram weergegeven die de regios en de gemiddelde huizenprijs per regio aangeeft. Klik er lekker doorheen!')

    # Create the basic figure
    fig = go.Figure()

    # Definieer een kleurenschema voor elke regio
    region_colors = {
        'Amsterdam-Noord': 'blue',
        'Amsterdam-West': 'red',
        'Amsterdam-Zuid': 'green',
        'Amsterdam Nieuw-West': 'orange',
        'Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)': 'purple',
        'Amsterdam-Oost': 'yellow',
        'Amsterdam-Zuid-Oost': 'cyan',
        'Oostelijk Havengebied (Amsterdam-Oost)': 'magenta',
        'Amsterdam Westpoort': 'lime'
    }

    # Loop through the different price ranges
    for PrijsRange in ['< €250k', '€250k - €500k', '€500k - €750k', '€750k - €1M', '€1M - €1,25M', '€1,25M - €1,5M', '> €1,5M']:
        # Subset the DataFrame
        df_PrijsRange = df[df.PrijsRange == PrijsRange]
        # Determine the color for each region
        colors = [region_colors[regio] for regio in df_PrijsRange['regio']]
        # Add a trace for each region subset
        fig.add_trace(go.Scatter(
            x=df_PrijsRange['vierkante_meter'],
            y=df_PrijsRange['prijzen'],
            mode='markers',
            name=PrijsRange,
            marker=dict(color=colors)))
    
    dropdown_buttons = [
        {'label': 'alle prijzen', 'method': 'update', 'args': [{'visible': [True, True, True, True, True, True, True]}, {'title':'alle prijzen'}]},
        {'label': '< €250k', 'method': 'update', 'args': [{'visible': [True, False, False, False, False, False, False]}, {'title':'< €250k'}]},
        {'label': '€250k - €500k', 'method': 'update', 'args': [{'visible': [False, True, False, False, False, False, False]}, {'title':'€250k - €500k'}]},
        {'label': '€500k - €750k', 'method': 'update', 'args': [{'visible': [False, False, True, False, False, False, False]}, {'title':'€500k - €750k'}]},
        {'label': '€750k - €1M', 'method': 'update', 'args': [{'visible': [False, False, False, True, False, False, False]}, {'title':'€750k - €1M'}]},
        {'label': '€1M - €1,25M', 'method': 'update', 'args': [{'visible': [False, False, False, False, True, False, False]}, {'title':'€1M - €1,25M'}]},
        {'label': '€1,25M - €1,5M', 'method': 'update', 'args': [{'visible': [False, False, False, False, False, True, False]}, {'title':'€1,25M - €1,5M'}]},
        {'label': '> €1,5M', 'method': 'update', 'args': [{'visible': [False, False, False, False, False, False, True]}, {'title':'> €1,5M'}]}
    ]

    # Update the figure to add dropdown menu
    fig.update_layout({
        'updatemenus':[{
                'type': 'dropdown',
                'x': 1, 'y': 1.1,
                'showactive': True, 'active': 0,
                'buttons': dropdown_buttons
                }]})

    # Plot weergeven met Streamlit
    st.plotly_chart(fig)

# Energielabel pagina
elif option == 'Energielabel':
    st.title("Energielabels")
    st.write("Hieronder zijn de hoeveelheid huizen met een bepaalde energielabel weergegeven.")

    # Create the basic figure
fig = go.Figure()

    # Loop through the states
for regio in [
        'Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)',
        'Oostelijk Havengebied (Amsterdam-Oost)',
        'Amsterdam-Noord',
        'Amsterdam Westpoort',
        'Amsterdam-West',
        'Amsterdam Nieuw-West',
        'Amsterdam-Zuid',
        'Amsterdam-Oost',
        'Amsterdam-Zuid-Oost']:

        # Subset the DataFrame
        df_sub = df[df.regio == regio]

        # Add a trace for each season
        fig.add_trace(go.Bar(x=df_sub["energielabel"], y=df_sub["energielabel"].value_counts(), name=regio))

    # add slider
sliders = [
        {'steps': [
            {'method': 'update', 'label': 'Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)',
             'args': [{'visible': [regio == 'Amsterdam-Centrum: Binnenstad (en delen van West en Westpoort)' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Oostelijk Havengebied (Amsterdam-Oost)',
             'args': [{'visible': [regio == 'Oostelijk Havengebied (Amsterdam-Oost)' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Amsterdam-Noord',
             'args': [{'visible': [regio == 'Amsterdam-Noord' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Amsterdam Westpoort',
             'args': [{'visible': [regio == 'Amsterdam Westpoort' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Amsterdam-West',
             'args': [{'visible': [regio == 'Amsterdam-West' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Amsterdam Nieuw-West',
             'args': [{'visible': [regio == 'Amsterdam Nieuw-West' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Amsterdam-Zuid',
             'args': [{'visible': [regio == 'Amsterdam-Zuid' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Amsterdam-Oost',
             'args': [{'visible': [regio == 'Amsterdam-Oost' for regio in df.regio]}]},
            {'method': 'update', 'label': 'Amsterdam-Zuid-Oost',
             'args': [{'visible': [regio == 'Amsterdam-Zuid-Oost' for regio in df.regio]}]}]}]

    # Update the figure to add sliders and show
fig.update_layout({'sliders': sliders, 'width': 800, 'height': 600})
st.plotly_chart(fig)

#vfkndfglkasdnf