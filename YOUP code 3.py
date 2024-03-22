import folium
from folium.plugins import MarkerCluster
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from ipywidgets import interact, widgets
import streamlit as st
from IPython.display import display
from streamlit_folium import st_folium
from PIL import Image
from streamlit_folium import folium_static

import pandas as pd

## Inhoudsopgave
st.sidebar.title('Inhoudsopgave')
option = st.sidebar.radio('Maak een keuze',['Intro','Correlaties', 'Voorspelling'])


#############  #  #   ##    ##   ####   ###      #############
#############  ####  #  #  #  #  # #    #  #     #############
#############  #  #   ##    ##   #      ###      #############

if option == 'Intro':

    # Pagina-titel en introductie
    st.title("startpagina")

    image = Image.open('vliegtuig.jpg')

    st.image(image, width= 680)

    # Welkomsttekst
    st.title('Welkom bij de vluchtanalysetool van Zurich')

    st.write("""
    Welkom bij deze vluchtanalysetool van Zurich! Deze webapp biedt u de mogelijkheid om diepgaande analyses uit te voeren op vluchtgegevens. 
    Of u nu geïnteresseerd bent in het verkennen van vluchthaven locaties , 
    het analyseren van vertragingen of het visualiseren van vluchtroutes op de kaart, 
    deze tool heeft alles wat u nodig heeft.

    ### Hoe werkt het?

    U kunt eenvoudig navigeren tussen verschillende analyses door te klikken op de gewenste inhoud aan de linkerkant van de pagina. 
    Elk tabblad die boven aan de pagina's te vinden zijn bieden specifieke functies om verschillende aspecten van het onderwerp die u geselecteerd hebt te analyseren. 
 

    ### Aan de slag!

    Om te beginnen hebben wij hieronder voor u al een wereldkaart gevisualiseerd, 
    selecteert u simpelweg een continent of land om de vluchten tussen zurich en u keuze te ontdekken. 
    Klik op de knoppen en start uw analyse!

    We wensen u veel plezier en succes bij het verkennen van de vluchtgegevens met deze tool!
    
    """)
             
    st.write("")
    st.write("")
             

    # Importeer de CSV-gegevens met de juiste delimiter ';'
    df = pd.read_csv('airports-extended-clean.csv', delimiter=';')

    # Importeer de CSV-gegevens voor de vluchtschema's met de juiste delimiter ','
    df_schedule_airport = pd.read_csv('schedule_airport.csv', sep=',')

    # Hernoem de kolom Org/Des om de datasets samen te voegen
    df_schedule_airport = df_schedule_airport.rename(columns={'Org/Des': 'ICAO'})

    # Samenvoegen van datasets op basis van de ICAO-code
    df_merged = df.merge(df_schedule_airport, on='ICAO')

    # Verwijder rijen met 'unknown', 'port' en 'station' type
    df_merged = df_merged[~df_merged['Type'].isin(['unknown', 'port', 'station'])]

    # Itereer over de unieke combinaties van breedte- en lengtegraden om dubbele locaties te voorkomen
    locations = df_merged[['Latitude', 'Longitude', 'Name', 'Country']].drop_duplicates()

    # Vervang komma's door punten in de kolommen 'Latitude' en 'Longitude'
    locations['Latitude'] = locations['Latitude'].astype(str).str.replace(',', '.')
    locations['Longitude'] = locations['Longitude'].astype(str).str.replace(',', '.')

    # Converteer de kolommen naar numerieke waarden
    locations['Latitude'] = pd.to_numeric(locations['Latitude'])
    locations['Longitude'] = pd.to_numeric(locations['Longitude'])

    # Maak de kaart aan
    mymap = folium.Map(location=[locations['Latitude'].mean(), locations['Longitude'].mean()], zoom_start=2)
    zurich_row = locations[locations['Name'] == 'Zürich Airport'].iloc[0]

    # Interactieve Filteren van Lijnen op Basis van Geselecteerde Landen
    def show_lines(selected_countries):
        filtered_map = folium.Map(location=[locations['Latitude'].mean(), locations['Longitude'].mean()], zoom_start=2)
        airport_cluster = MarkerCluster().add_to(filtered_map)
        for idx, row in locations.iterrows():
            if row['Country'] in selected_countries:
                folium.PolyLine(locations=[[float(zurich_row['Latitude']), float(zurich_row['Longitude'])], [float(row['Latitude']), float(row['Longitude'])]], color='blue').add_to(filtered_map)

        for idx, row in locations.iterrows():
            if row['Country'] in selected_countries:
                latitude = float(row['Latitude'])  # Omzetten naar een kommagetal
                longitude = float(row['Longitude'])  # Omzetten naar een kommagetal
                popup_content = folium.Popup(row['Name'], parse_html=True)
                icon = folium.Icon(color='blue', icon="plane")
                folium.Marker([latitude, longitude], popup=popup_content, icon=icon).add_to(airport_cluster) 
        
        return filtered_map


    # Interactieve widget om landen te selecteren
    selected_countries = st.multiselect('Selecteer de landen die je wil zien:', options=locations['Country'].unique(), default=locations['Country'].unique())

    # Voeg de titel boven de kaart toe
    st.markdown("<h1 style='text-align: center;'>Vluchten van/naar Zurich</h1>", unsafe_allow_html=True)

    # Interactie met de kaart
    folium_static(show_lines(selected_countries))

elif option == 'Correlaties':
    st.title("Correlatie")

    airport_schedules = pd.read_csv('schedule_airport.csv')

    nieuwe_kolomnamen = {
        'STD': 'Datum',
        'FLT': 'Vlucht_nummer',
        'STA_STD_ltc': 'Geplande_aankomst',
        'ATA_ATD_ltc': 'Werkelijke_aankomst',
        'LSV': 'Aankomend_L_Weggaand_S',
        'TAR': 'Geplande_gate',
        'GAT': 'Werkelijke_gate',
        'DL1': 'DL1',
        'IX1': 'IX1',
        'DL2': 'DL2',
        'IX2': 'IX2',
        'ACT': 'Vliegtuig_type',
        'RWY': 'Landingsbaan_Startbaan',
        'RWC': 'RWC',
        'Identifier': 'Flight_Identifier',
        'Org/Des': 'Van_Naar'
    }

    # Hernoem de kolomnamen met behulp van het woordenboek
    airport_schedules = airport_schedules.rename(columns=nieuwe_kolomnamen)
    airport_schedules = airport_schedules.drop(columns= ['DL1','IX1','DL2','IX2'])

    # Filteren van de DataFrame op vluchten naar Canada en een kopie maken
    extended_data = pd.read_csv('airports-extended-clean.csv', sep=';')

    # Vervang komma's door punten in Latitude en Longitude kolommen
    extended_data['Latitude'] = extended_data['Latitude'].astype(str).str.replace(',', '.')
    extended_data['Longitude'] = extended_data['Longitude'].astype(str).str.replace(',', '.')

    selected_country_extended_data = extended_data[extended_data['Country'].isin(['Spain', 'Italy', 'France'])]
    selected_country_extended_data = selected_country_extended_data[selected_country_extended_data['Type'] == 'airport']

    combie_df = pd.merge(airport_schedules, selected_country_extended_data, left_on='Van_Naar', right_on='ICAO', how='inner')


    ########## Extended data ###################
    # Filteren van de DataFrame op vluchten naar Canada en een kopie maken
    extended_data = pd.read_csv('airports-extended-clean.csv', sep=';')

    # Vervang komma's door punten in Latitude en Longitude kolommen
    extended_data['Latitude'] = extended_data['Latitude'].astype(str).str.replace(',', '.')
    extended_data['Longitude'] = extended_data['Longitude'].astype(str).str.replace(',', '.')

    canada_flights = extended_data[extended_data['Country'] == 'Canada'].copy()

    ########## Schedules data ###################
    airport_schedules = pd.read_csv('schedule_airport.csv')

    nieuwe_kolomnamen = {
        'STD': 'Datum',
        'FLT': 'Vlucht_nummer',
        'STA_STD_ltc': 'Geplande_aankomst',
        'ATA_ATD_ltc': 'Werkelijke_aankomst',
        'LSV': 'Aankomend_L_Weggaand_S',
        'TAR': 'Geplande_gate',
        'GAT': 'Werkelijke_gate',
        'DL1': 'DL1',
        'IX1': 'IX1',
        'DL2': 'DL2',
        'IX2': 'IX2',
        'ACT': 'Vliegtuig_type',
        'RWY': 'Landingsbaan_Startbaan',
        'RWC': 'RWC',
        'Identifier': 'Flight_Identifier',
        'Org/Des': 'Van_Naar'
    }

    # Hernoem de kolomnamen met behulp van het woordenboek
    airport_schedules = airport_schedules.rename(columns=nieuwe_kolomnamen)
    airport_schedules = airport_schedules.drop(columns=['DL1', 'IX1', 'DL2', 'IX2'])

    merged_df = pd.merge(airport_schedules, extended_data, left_on='Van_Naar', right_on='ICAO', how='inner')

    ########## Merged data ###################
    # Merge de dataframes op basis van een gemeenschappelijke kolom
    merged_df['Timezone'] = pd.to_numeric(merged_df['Timezone'], errors='coerce')
    merged_df['Datum'] = pd.to_datetime(merged_df['Datum'])

    # Stap 1: Bereken de vertraging
    merged_df['Vertraging'] = pd.to_datetime(merged_df['Werkelijke_aankomst']) - pd.to_datetime(merged_df['Geplande_aankomst']) - pd.to_timedelta(merged_df['Timezone'], unit='h')

    Aankomst_df = merged_df[merged_df['Aankomend_L_Weggaand_S'] == 'L']
    Vertrek_df = merged_df[merged_df['Aankomend_L_Weggaand_S'] == 'S']

    # Stap 2: Bekijk de statistieken van de vertragingen
    Aankomst_vertraging_stats = Aankomst_df['Vertraging'].describe()
    print("Statistieken van vertragingen:")
    print(Aankomst_vertraging_stats)

    plt.figure(figsize=(10, 6))
    plt.hist(Aankomst_df['Vertraging'].dt.total_seconds() / 60, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Vertraging (minuten)')
    plt.ylabel('Frequentie')
    plt.title('Verdeling van vertragingen')
    plt.grid(True)
    st.pyplot(plt)

    Vertrek_vertraging_stats = Vertrek_df['Vertraging'].describe()
    print("Statistieken van vertragingen:")
    print(Vertrek_vertraging_stats)

    plt.figure(figsize=(10, 6))
    plt.hist(Vertrek_df['Vertraging'].dt.total_seconds() / 60, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Vertraging (minuten)')
    plt.ylabel('Frequentie')
    plt.title('Verdeling van vertragingen')
    plt.grid(True)
    st.pyplot(plt)

    import pandas as pd
    import plotly.graph_objs as go
    import streamlit as st


    # Zorg ervoor dat de 'Datum'-kolom correct wordt omgezet naar een datetime-datatype
    combie_df['Datum'] = pd.to_datetime(combie_df['Datum'])

    # Functie om het lijndiagram te maken en weer te geven op basis van de geselecteerde naam
    # Functie om het lijndiagram te maken en weer te geven
    def create_line_chart(dataframe):
        # Groepeer de gegevens op datum en tel het aantal vluchten per dag
        grouped_df = dataframe.groupby(dataframe['Datum'].dt.date).size().reset_index(name='Aantal_vliegtuigen')


        # Maak het lijndiagram
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grouped_df['Datum'], y=grouped_df['Aantal_vliegtuigen'],
                        line_shape='hvh'))
            
        # Titel en labels
        fig.update_layout(title='Aantal vliegtuigen op Zürich', xaxis_title='Datum', yaxis_title='Aantal vliegtuigen')

        # Add range slider
        fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(step="all"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=7, label="1W", step="day", stepmode="backward"), 
                            dict(count=1, label="1d", step="day", stepmode="backward")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )

        return fig

    # Weergeef de grafiek
    fig = create_line_chart(combie_df)
    st.plotly_chart(fig)

     ########## Correlatiematrix ###################

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Lees het CSV-bestand in een DataFrame met naam omdat we verschillend gaan kijken naar dezelfde data
    schedule_airport_camiel = pd.read_csv('schedule_airport.csv')
    airports_extended_camiel = pd.read_csv('airports-extended-clean.csv', sep=';')

    # Vind de unieke waarden
    unieke_waarden_gate = schedule_airport_camiel['TAR'].unique()

    print(unieke_waarden_gate)

    unieke_waarden_gate_exact = schedule_airport_camiel['GAT'].unique()

    print(unieke_waarden_gate_exact)

    # Maak een DataFrame voor df_gate
    df_gate = pd.DataFrame(unieke_waarden_gate_exact, columns=['Waarde'])

    # Maak een DataFrame voor df_geplande_gate
    df_geplande_gate = pd.DataFrame(unieke_waarden_gate, columns=['Waarde'])

    # Combineer de waarden van beide DataFrames om unieke waarden te krijgen
    alle_waarden = pd.concat([df_gate['Waarde'], df_geplande_gate['Waarde']], ignore_index=True)

    # Factorize de gecombineerde waarden en houd de mapping
    labels, unieke_waarden = pd.factorize(alle_waarden)

    # Wijs de labels toe aan de gate kolom in df_gate
    df_gate['gate'] = labels[:len(df_gate)]

    # Wijs de labels toe aan de geplande_gate kolom in df_geplande_gate
    df_geplande_gate['geplande_gate'] = labels[len(df_gate):]

    # Print beide DataFrames
    print("df_gate:")
    print(df_gate)
    print("df_geplande_gate:")
    print(df_geplande_gate)

    schedule_airport_camiel.rename(columns={'STA_STD_ltc': 'plan_tijd'}, inplace=True)
    schedule_airport_camiel.rename(columns={'ATA_ATD_ltc': 'real_tijd'}, inplace=True)

    # Converteer de kolommen naar datetime-objecten
    schedule_airport_camiel['real_tijd'] = pd.to_datetime(schedule_airport_camiel['real_tijd'])
    schedule_airport_camiel['plan_tijd'] = pd.to_datetime(schedule_airport_camiel['plan_tijd'])

    # Bereken het verschil in minuten
    schedule_airport_camiel['verschil_minuten'] = (schedule_airport_camiel['real_tijd'] - schedule_airport_camiel['plan_tijd']).dt.total_seconds() / 60

    # Rond kolom af op twee decimalen voor duidelijkheid
    schedule_airport_camiel['verschil_minuten'] = schedule_airport_camiel['verschil_minuten'].round(2)

    # Maak een compact DataFrame met de gewenste kolommen
    schedule_airport_camiel_compact = schedule_airport_camiel.loc[:, ['Org/Des', 'LSV', 'RWY', 'verschil_minuten']]

    # Vind de unieke waarden
    unieke_waarden_tz = airports_extended_camiel['Tz'].str.split('/').str[0].unique()

    print(unieke_waarden)

    # Gebruik str.split() om de string te splitsen op '/'
    # Selecteer het eerste deel (vóór de '/')
    # Maak er een categorische variabele van
    schedule_airport_camiel_compact['regio'] = airports_extended_camiel['Tz'].str.split('/').str[0]
    schedule_airport_camiel_compact['regio'] = schedule_airport_camiel_compact['regio'].astype('category')

    # Wijs numerieke labels toe aan de categorische variabele
    schedule_airport_camiel_compact['regio_code'] = schedule_airport_camiel_compact['regio'].cat.codes

    # Toon het compacte DataFrame
    schedule_airport_camiel_compact.head()

    # Maak er een categorische variabele van
    schedule_airport_camiel_compact['stad'] = schedule_airport_camiel_compact['Org/Des'].astype('category')

    # Wijs numerieke labels toe aan de categorische variabele
    schedule_airport_camiel_compact['stad_code'] = schedule_airport_camiel_compact['stad'].cat.codes

    # Maak er een categorische variabele van
    schedule_airport_camiel_compact['in_out'] = schedule_airport_camiel_compact['LSV'].astype('category')

    # Wijs numerieke labels toe aan de categorische variabele
    schedule_airport_camiel_compact['IN_OUT'] = schedule_airport_camiel_compact['in_out'].cat.codes

    schedule_airport_camiel_correlatie = schedule_airport_camiel_compact.loc[:, ['verschil_minuten','RWY', 'regio_code', 'stad_code', 'IN_OUT']]

    schedule_airport_camiel_correlatie['gate'] = df_gate['gate']
    schedule_airport_camiel_correlatie['geplande_gate'] = df_geplande_gate['geplande_gate']

    # Bereken de correlatiematrix
    correlatie_matrix = data.corr()

    # Definieer de Streamlit-applicatie
    def main():
        st.title("Correlatiematrix Visualisatie")

    # Toon de dataset
    st.subheader("Dataset")
    st.dataframe(data)

    # Plot de correlatiematrix met behulp van seaborn
    st.subheader("Correlatiematrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlatie_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlatiematrix')
    st.pyplot()

    # Voer de applicatie uit
    if __name__ == "__main__":
        main()

     ########## Plot 1 ###################

    # Maak een dictionary om de mapping tussen regio's en regiocodes vast te leggen
    region_mapping = dict(zip(schedule_airport_camiel_correlatie['regio_code'], schedule_airport_camiel_compact['regio']))

    # Print de mapping
    for code, region in region_mapping.items():
        print(f"regio code {code} komt overeen met regio '{region}'")


    # Groepeer de gegevens per regiocode en bereken het gemiddelde vluchtverschil per regio
    gemiddeld_per_regio = schedule_airport_camiel_correlatie.groupby('regio_code')['verschil_minuten'].mean()

    # Maak een nieuwe DataFrame met de gemiddelde vertraging per regio
    gemiddelde_vertraging_per_regio = pd.DataFrame({'regio_code': gemiddeld_per_regio.index, 'gemiddelde_vertraging': gemiddeld_per_regio.values})

    # Rond de gemiddelde vertraging af op 2 decimalen
    gemiddelde_vertraging_per_regio['gemiddelde_vertraging'] = gemiddelde_vertraging_per_regio['gemiddelde_vertraging'].round(2)

    # Voeg de regio's toe aan de DataFrame vertraging_per_regio_df met behulp van de region_Code
    gemiddelde_vertraging_per_regio['regio'] = gemiddelde_vertraging_per_regio['regio_code'].map(region_mapping)

    # Vervang 'nan' door een lege string
    gemiddelde_vertraging_per_regio['regio'] = gemiddelde_vertraging_per_regio['regio'].replace('nan', np.nan)

    # Verwijder NaN-waarden en '\N'-waarden uit de 'Regio'-kolom
    gemiddelde_vertraging_per_regio = gemiddelde_vertraging_per_regio.dropna(subset=['regio'])
    gemiddelde_vertraging_per_regio = gemiddelde_vertraging_per_regio[gemiddelde_vertraging_per_regio['regio'] != '\\N']

    # Creëer een figuur en een set subplots
    fig, ax = plt.subplots()

    # Plot de gemiddelde vertraging per regio als staafdiagram
    ax.bar(gemiddelde_vertraging_per_regio['regio'], gemiddelde_vertraging_per_regio['gemiddelde_vertraging'], color='purple')

    # Voeg labels toe
    ax.set_title('Gemiddelde Vertraging per regio')
    ax.set_xlabel('Regio')
    ax.set_ylabel('Gemiddelde Vertraging (minuten)')

    # Roteer de x-labels voor een betere leesbaarheid
    plt.xticks(rotation=45, ha='right')

    # Toon het staafdiagram
    plt.tight_layout()
    st.pyplot(fig)

    ########## Plot 2 ###################

    # Lees het CSV-bestand in een DataFrame met naam omdat we verschillend gaan kijken naar dezelfde data
    schedule_airport_camiel_plot = pd.read_csv('schedule_airport.csv')
    airports_extended_camiel_plot = pd.read_csv('airports-extended-clean.csv', sep=';')

    #hernoemen kolom Org/Des om zo de datasets samen te voegen
    schedule_airport_camiel_plot = schedule_airport_camiel_plot.rename(columns={'Org/Des': 'ICAO'})

    # Merge-operatie op de sleutelkolom 'Key'
    merged_schedule_airport_camiel_plot = pd.merge(airports_extended_camiel_plot, schedule_airport_camiel_plot, on='ICAO', how='inner')

    #kolommen omzetten naar tijd

    schedule_airport_camiel_plot['STA_STD_ltc'] = pd.to_datetime(schedule_airport_camiel_plot['STA_STD_ltc'])

    schedule_airport_camiel_plot['ATA_ATD_ltc'] = pd.to_datetime(schedule_airport_camiel_plot['ATA_ATD_ltc'])
 
    #kolom toevoegen

    schedule_airport_camiel_plot['verschil_in_seconden'] = (schedule_airport_camiel_plot['STA_STD_ltc'] - schedule_airport_camiel_plot['ATA_ATD_ltc']).dt.total_seconds()

    #string to integer

    schedule_airport_camiel_plot['verschil_in_seconden_str'] = schedule_airport_camiel_plot['verschil_in_seconden'].astype(str)
 
    # Voeg een nieuwe kolom toe voor de eerste teken

    schedule_airport_camiel_plot['first_token'] = schedule_airport_camiel_plot['verschil_in_seconden_str'].str.slice(0, 1)
 
    #kijken of een vliegtuig vertraging heeft ja of nee

    schedule_airport_camiel_plot['Delay?'] = np.where(schedule_airport_camiel_plot['first_token'] == '-', 'Yes', 'No')

    # Gebruik str.split() om de string te splitsen op '/'
    # Selecteer het eerste deel (vóór de '/')
    # Maak er een categorische variabele van
    schedule_airport_camiel_plot['regio'] = airports_extended_camiel_plot['Tz'].str.split('/').str[0]
    schedule_airport_camiel_plot['regio'] = schedule_airport_camiel_plot['regio'].astype('category')

    # Groepeer de gegevens per regio en vertraging en tel het aantal vluchten
    schedule_airport_camiel_plot2 = schedule_airport_camiel_plot.groupby(['regio', 'Delay?']).size().unstack(fill_value=0)

    # Plot het gestapelde staafdiagram
    fig, ax = plt.subplots()
    schedule_airport_camiel_plot2.plot(kind='bar', stacked=True, color=['purple', 'pink'], ax=ax)

    # Voeg labels toe
    ax.set_title('Aantal vertraagde en niet-vertraagde vluchten per regio')
    ax.set_xlabel('Regio')
    ax.set_ylabel('Aantal vluchten')
    plt.xticks(rotation=45, ha='right')

    # Toon het staafdiagram
    plt.tight_layout()

    # Toon het staafdiagram in Streamlit
    st.pyplot(fig)

    ########## Plot 3 ###################



else:
    st.title("Voorspelling")

    import pandas as pd
    import folium
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import seaborn as sns

    df_schedule_airport = pd.read_csv('schedule_airport.csv', sep=',')
    df= pd.read_csv('airports-extended-clean.csv', sep=';')

    #hernoemen kolom Org/Des om zo de datasets samen te voegen
    df_schedule_airport = df_schedule_airport.rename(columns={'Org/Des': 'ICAO'})

    #samenvoegen van datasets
    df_merged = df.merge(df_schedule_airport, on= 'ICAO')

    #kolommen omzetten naar tijd
    df_merged['STA_STD_ltc'] = pd.to_datetime(df_merged['STA_STD_ltc'])
    df_merged['ATA_ATD_ltc'] = pd.to_datetime(df_merged['ATA_ATD_ltc'])

    #kolom toevoegen
    df_merged['verschil_in_seconden'] = (df_merged['ATA_ATD_ltc'] - df_merged['STA_STD_ltc']).dt.total_seconds()

    #onderscheidt maken tussen inbound en outbound om zou ...
    df_schedule_airport_L = df_merged[df_merged['LSV'] == 'L']
    df_schedule_airport_S = df_merged[df_merged['LSV'] == 'S']

    european_countries = [
    'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 
    'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 
    'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 
    'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 
    'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 
    'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine', 
    'United Kingdom', 'Vatican City'
    ]

    df_schedule_airport_L_europa = df_schedule_airport_L[df_schedule_airport_L['Country'].isin(european_countries)]

    #string to integer
    df_schedule_airport_L_europa['verschil_in_seconden'] = df_schedule_airport_L_europa['verschil_in_seconden'].astype(str)

    # Voeg een nieuwe kolom toe voor de eerste teken
    df_schedule_airport_L_europa['first_token'] = df_schedule_airport_L_europa['verschil_in_seconden'].str.slice(0, 1)

    #kijken of een vliegtuig vertraging heeft ja of nee
    df_schedule_airport_L_europa['Delay?'] = np.where(df_schedule_airport_L_europa['first_token'] == '-', 'No', 'Yes')

    # Maak nieuwe dataframe waar alleen de inbound vluchten binnen europa 
    # meegenomen worden die vertraging hebben
    df_schedule_airport_L_europa_delay = df_schedule_airport_L_europa[df_schedule_airport_L_europa['Delay?'] == 'Yes']

    # Bekijk de nieuwe DataFrame
    df_schedule_airport_L_europa_delay.head()

    # Bekijk de lengte
    len(df_schedule_airport_L_europa_delay)

    # alle komma's naar punten veranderen om zo de hemelsbreedte te kunnen berekenen
    df_schedule_airport_L_europa_delay = df_schedule_airport_L_europa_delay.applymap(lambda x: str(x).replace(',', '.'))
    #df_schedule_airport_S_europa_delay = df_schedule_airport_S_europa_delay.applymap(lambda x: str(x).replace(',', '.'))

    ### berekenen voor alle inbound vluchten ###

### berekenen voor alle inbound vluchten ###


    #Zorg dat het bestand vinc.py in dezelfde map zit als het bestand waar je in werkt
    import vinc
    #importeer de functie v_direct met de onderstaande code
    from vinc import v_direct

    #naar float omzetten om er mee te kunnen rekenen
    df_schedule_airport_L_europa_delay_float = df_schedule_airport_L_europa_delay[['Latitude', 'Longitude']].applymap(lambda x: float(str(x).replace(',', '.')))

    # Definieer de coördinaten van Zürich
    zurich_coords = (47.46469879, 8.54916954)

    # Definieer een functie om de afstand te berekenen voor elke rij in de dataset
    def bereken_afstand(row):
        # Coördinaten van het punt van de huidige rij
        punt_coords = (row['Latitude'], row['Longitude'])
        # Bereken de afstand tussen het huidige punt en Zürich
        afstand = v_direct(zurich_coords, punt_coords)
        return afstand

    # Voeg een nieuwe kolom 'Afstand_tot_Zurich' toe aan je dataset
    # met de berekende afstanden
    df_schedule_airport_L_europa_delay['Afstand_tot_Zurich'] = df_schedule_airport_L_europa_delay_float.apply(bereken_afstand, axis=1)

    # Toon de dataset met de nieuwe kolom
    df_schedule_airport_L_europa_delay.head()

        #from pipetorch import DFrame
    #from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import statsmodels.api as sm
    import seaborn as sns
    import statsmodels.formula.api as smf
    from sklearn.metrics import mean_squared_error

        # Stap 1: Dataset laden
    #excel bestand naar csv omzetten
    #boven al gedefinieerd

    #rijen met missende waarden eruithalen
    df_schedule_airport_L_europa_delay = df_schedule_airport_L_europa_delay.dropna()

    # Verwijders rijen waarin de waarde van 'city' is niet Zurich
    filtered_df = df_schedule_airport_L_europa_delay.loc[df_schedule_airport_L_europa_delay['City'] != 'Zurich']
    filtered_df['verschil_in_seconden'] = filtered_df['verschil_in_seconden'].astype(float)

    # Verwijder rijen waarin de waarde van 'col1' gelijk is aan 0
    #filtered_df = filtered_df[filtered_df['verschil_in_seconden'] != 0.0]
    print(len(filtered_df))

    #groupby Name, mean van verschil_in_seconden
    gemiddelde_verschil = filtered_df.groupby('Name')[['verschil_in_seconden', 'Afstand_tot_Zurich']].mean()

    #dataset sorteren
    df_sorted = gemiddelde_verschil.sort_values(by='verschil_in_seconden')
    df_sorted['Afstand_tot_Zurich_km'] = (df_sorted['Afstand_tot_Zurich']/1000)

    #types veranderen naar floats
    df_sorted['Afstand_tot_Zurich_km'] = pd.to_numeric(df_sorted['Afstand_tot_Zurich_km'], errors='coerce')
    df_sorted['verschil_in_seconden'] = pd.to_numeric(df_sorted['verschil_in_seconden'], errors='coerce')
    df_sorted['Afstand_tot_Zurich'] = pd.to_numeric(df_sorted['Afstand_tot_Zurich'], errors='coerce')
    df_sorted.dtypes

        # Gebruik lmplot om lineaire regressie toe te passen
    sns.lmplot(x='Afstand_tot_Zurich_km', y='verschil_in_seconden', data=df_sorted)

    # Optioneel: voeg een titel toe
    plt.title('Lineaire regressie tussen Afstand tot Zurich (km) en Verschil in seconden')

    # Toon de plot
    st.pyplot()

        # Stap 3: Lineaire regressie toepassen
    X = df_sorted['Afstand_tot_Zurich_km']  # Selecteer de predictor variabele(n)
    y = df_sorted['verschil_in_seconden']  # Selecteer de responsvariabele

    model = smf.ols(formula = 'y ~ X', data = df_sorted).fit()
    model.summary()

        # Reshape X and y if necessary
    if X.ndim == 1:
        X = X.values.reshape(-1, 1)
    if y.ndim == 1:
        y = y.values.reshape(-1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate RMSE
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

    print("Training RMSE:", rmse_train)
    print("Testing RMSE:", rmse_test)