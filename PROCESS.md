# What was the process in creating the AI

## 1. Projekt aufsetzten

WSL2 einrichten, um Tensorflow auf Windows mit PyCharm zu benutzen. Hierzu wurde einfach WSL2 nach Microsofts
dokumentation installiert. Hiermit gab es keine Probleme

## 2. Mit Tensorflow verwand werden

Erst wurde ein simples Projekt aufgesetzt bei dem das Tutorial von Tensorflow gefolgt wurde.
Um dieses richtig zu verstehen, wurden die einzelnen Komponenten und Schritte ausführlich zu einem späteren Zeitpunkt
ohne Tutorial beschrieben. Es wurde auch kurz eine Exkursion gemacht, das trainierte Model zu speichern und wieder zu
laden, um nicht das Model immer neu trainieren zu müssen.

## 3. Projekt Auswahl

Für sehr lange war unklar, was für ein spezifisches Projekt gemacht werden soll. Auf dem Tisch stand:
- Image klassifizierung von noch unbestimmten Objekten, dies fühlte sich jedoch mit dem Tensorflow framework sehr
  einfach, da dies zum Großteil nur das Sammeln des Trainingsmaterials ist und ein wenig Anpassen des Tutorial models.
- Ein Videospiel lerne mithilfe eines evolutionären Algorithmus. Hierfür kamen infrage Snake und Connect 4.

Nach etwas Diskussion mit ein paar unabhängigen Kommilitonen über die möglichen Vor- und Nachteile von Snake und
Connect 4 wurde sich entschieden Connect 4 durch einen evolutionären Algorithmus dem Model beizubringen. Diese
Entscheidung wurde getroffen, da das Übergeben des Spielstandes and das Model sich einfacher darstellen lässt.
Probleme bei Snake wäre das Enkodieren der vorherigen Spielzustände um herauszufinden in welche Richtung sich gerade
bewegt wird (Letztlich wäre eine mögliche Lösung das Alter der Schlang in einem Feld in dem 2D Bild des Spielzustandes
zu enkodieren z.B. je länger auf einem Feld, umso höher dessen Wert).

## 4. Implementing Connect 4

Das Spielbrett besteht aus einem 2 dimensionalen 7x6 array. Hierbei wird ein leeres Feld als 0 dargestellt, ein Feld von
Spieler 1 als 1 und ein Feld von Spieler 2 als 2. Diese wechseln sich immer ab die `Connect4.place` Methode aufzurufen,
dem sie die Spalte übergeben. Anders als normales Connect 4 verliert man automatisch, wenn man versuch in eine volle
Spalte sein Zug zu machen (das Einfügen ist sehr simpel: Gehe alle Zeilen rückwerts durch und such die erste Zeile, in
der die angegebene Spalte 0 ist, gibt es kein leeres Feld hat man verloren). Das Verlieren beim setzten außerhalb
valider Felder ist so implementiert, um hoffentlich beim Model zu verhindern, in volle Spalten zu spielen.

Das Überprüfen, ob gewonnen wurde, geschieht nach dem Zug und geschieht, indem eine Kopie des Brettes, wo nur die Felder
des Spielers zu 1 gesetzt werden (alles andere ist 0), worauf dann mit 4 verschiedenen Kernel (für vertikal, horizontal
und beide Diagonalen) eine Konvolution gemacht wird. Die resultierenden Bilder enthalten Werte von 0 bis 4 und wenn eine
4 enthalten ist, heißt es, dass alle Werte eines Kernel einen Wert im Bild hatten und somit gewonnen wurde.

## 5. Implementing the Neural Network

### Fitness funktion

Für die Fitness funktion der Modelle werden einfach die Zuganzahl und ob gewonnen wurde oder nicht einbezogen. Es wird
auf einen maximalen Fitnessscore gezielt. Hierbei wird der Fitness score für einen Sieg mit
`(ROW_COUNT * COL_COUNT) / turns` berechnet, während ein loss mit `turns / (ROW_COUNT * COL_COUNT)` berechnet wird.
Dies versichert, dass ein Sieg immer > 1 ergibt, währen eine Niederlage immer < 1. Theoretisch ist das Maximum an score
6 Punkte, indem immer in einer Spalte gespielt wird und man außerhalb am siebten Zug spielt.

### Network Model

Dies sollte in etwa wie folgt aussehen:
```
tf.keras.models.Sequential([
	tf.keras.layers.Input(shape=(ROW_COUNT, COL_COUNT)),
	tf.keras.layers.Conv2D(?, kernel_size=(4, 4)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(ROW_COUNT, activation=tf.keras.activations.softmax),
])
```
Jedoch nach etwas recherche hatte ich gesehen, dass möglicherweise mehrere kleinere Kernel besser funktionieren können.
Es haben sich leider Probleme einbinden der `Conv2D` layer ergeben, die sich nach extensiver Recherche und Debugging
nicht lösen ließen.

### Der evolutionäre Algorithmus

Es werden zu begin `POPULATION` an zufälligen Modellen erstellt. Diese spielen dann alle einen zufälligen Gegner (dies
wurde später durch einen rein zufälligen Algorithmus, der eine valide Spalte auswählt ersetzt. Eigentlich sollte dies
durch einen Minimax algorithmus ersetzt werden, jedoch habe ich die Implementation nie zum Funktionieren bekommen).

Mit diesen Spielen gegen Gegner werden ihnen einen Fitnesswert wie oben beschrieben zugeteilt. Diese werden dann
aufsteigend sortiert und die bessere Hälfte wird in die nächste Generation übernommen. Die andere Hälfte wird durch
das Kreuzen von 2 zufälligen Eltern der übernommenen Population ersetzt. Beim Kreuzen werden die Weights zufällig von
einem Elternteil ausgewählt und nachdem das Model besteht werden die einzelnen Weights, wenn sie eine bestimmte
Wahrscheinlichkeit zutrifft, um einen zufälligen Wert um der Standard Normalverteilung positive oder negative mutiert.
