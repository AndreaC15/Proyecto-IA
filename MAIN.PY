# Importaciones para la creación de API
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Configuración de la ruta de datos de NLTK
nltk.data.path.append('C:/Users/SILVIA/AppData/Roaming/nltk_data')

# Configuración inicial de NLTK - Descarga de paquetes necesarios
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Paquetes de NLTK descargados correctamente")
except Exception as e:
    print(f"Error al descargar paquetes de NLTK: {e}")

# ETAPA 4: Cargar y procesar el dataset de Netflix
def cargar_dataset_netflix(ruta_archivo="DATASET/netflix_titles.csv"):
    """
    Carga el dataset de Netflix desde un archivo CSV con solo las columnas necesarias
    """
    # Columnas específicas que necesitamos
    columnas_necesarias = ['show_id', 'title', 'type', 'release_year', 'listed_in', 'description']
    
    try:
        df = pd.read_csv(ruta_archivo, usecols=columnas_necesarias)
        print(f"Dataset cargado exitosamente: {len(df)} registros")
        print(f"Columnas cargadas: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_archivo}")
        return None
    except KeyError as e:
        print(f"Error: Una o más columnas no existen en el dataset: {e}")
        print(f"Columnas solicitadas: {columnas_necesarias}")
        return None
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None


def identificar_columnas_dataset(df):
    """
    Función específica para identificar y analizar todas las columnas del dataset
    """
    if df is None:
        print("No hay dataset para analizar")
        return
    
    print("\n" + "="*60)
    print("ANÁLISIS DETALLADO DE COLUMNAS DEL DATASET")
    print("="*60)
    
    # Información básica
    print(f"\n📊 RESUMEN GENERAL:")
    print(f"   • Total de columnas: {len(df.columns)}")
    print(f"   • Total de registros: {len(df)}")
    
    # Análisis por tipo de dato
    tipos_datos = df.dtypes.value_counts()
    print(f"\n📋 DISTRIBUCIÓN POR TIPOS DE DATOS:")
    for tipo, cantidad in tipos_datos.items():
        print(f"   • {tipo}: {cantidad} columnas")
    
    # Lista detallada de columnas
    print(f"\n📝 LISTA DETALLADA DE COLUMNAS:")
    for i, columna in enumerate(df.columns, 1):
        tipo_dato = df[columna].dtype
        valores_nulos = df[columna].isnull().sum()
        valores_unicos = df[columna].nunique()
        
        print(f"\n   {i:2d}. {columna}")
        print(f"       • Tipo de dato: {tipo_dato}")
        print(f"       • Valores nulos: {valores_nulos}")
        print(f"       • Valores únicos: {valores_unicos}")
        
        # Mostrar algunos valores de ejemplo para columnas de texto
        if tipo_dato == 'object' and valores_unicos > 0:
            valores_ejemplo = df[columna].dropna().head(3).tolist()
            print(f"       • Ejemplos: {valores_ejemplo}")
    
    # Identificar columnas específicas importantes
    print(f"\n🎯 COLUMNAS CLAVE IDENTIFICADAS:")
    columnas_clave = []
    
    for columna in df.columns:
        columna_lower = columna.lower()
        if any(palabra in columna_lower for palabra in ['title', 'name', 'titulo', 'nombre']):
            columnas_clave.append(f"📺 {columna} (Título/Nombre)")
        elif any(palabra in columna_lower for palabra in ['description', 'descripcion', 'summary', 'resumen']):
            columnas_clave.append(f"📝 {columna} (Descripción)")
        elif any(palabra in columna_lower for palabra in ['genre', 'genero', 'category', 'categoria']):
            columnas_clave.append(f"🏷️ {columna} (Género/Categoría)")
        elif any(palabra in columna_lower for palabra in ['year', 'año', 'date', 'fecha']):
            columnas_clave.append(f"📅 {columna} (Fecha/Año)")
        elif any(palabra in columna_lower for palabra in ['rating', 'calificacion', 'score']):
            columnas_clave.append(f"⭐ {columna} (Calificación)")
        elif any(palabra in columna_lower for palabra in ['director', 'director']):
            columnas_clave.append(f"🎬 {columna} (Director)")
        elif any(palabra in columna_lower for palabra in ['cast', 'reparto', 'actor']):
            columnas_clave.append(f"👥 {columna} (Reparto)")
    
    if columnas_clave:
        for columna_clave in columnas_clave:
            print(f"   {columna_clave}")
    else:
        print("   No se identificaron columnas clave específicas")
    
    print("\n" + "="*60)

# ETAPA 5: Inicialización de la API
app = FastAPI(
    title="mi aplicacion de peliculas",
    version="1.0.0",
    description="API para análisis y recomendación de películas usando procesamiento de lenguaje natural"
)

# Variable global para almacenar el dataset
dataset_netflix = None

# Cargar el dataset al iniciar (si está disponible)
print("\n=== CARGA DEL DATASET NETFLIX ===")
try:
    dataset_netflix = cargar_dataset_netflix()
    if dataset_netflix is None:
        print("⚠️ Advertencia: Dataset no disponible. Algunas funcionalidades pueden no funcionar.")
except Exception as e:
    print(f"⚠️ Advertencia: Error al cargar dataset: {e}")
    dataset_netflix = None

# ETAPA 6: Creación de rutas de la API

# RUTA INICIAL: Mensaje de bienvenida con chatbot
@app.get("/", response_class=HTMLResponse)
async def ruta_inicial():
    """
    Ruta inicial que muestra la interfaz del chatbot para buscar películas
    """
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot de Películas - Netflix</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            
            body {
                font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #e50914 100%);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            .container {
                max-width: 900px;
                width: 100%;
                background: #1a1a1a;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(229, 9, 20, 0.4);
                overflow: hidden;
                animation: slideIn 0.5s ease-out;
                border: 2px solid #e50914;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .header {
                background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.8em;
                margin-bottom: 10px;
                font-weight: 700;
                letter-spacing: -0.5px;
                text-transform: uppercase;
            }
            
            .header p {
                font-size: 1.15em;
                opacity: 0.95;
                font-weight: 400;
                letter-spacing: 0.5px;
            }
            
            .chat-container {
                padding: 30px;
                height: 500px;
                overflow-y: auto;
                background: #0a0a0a;
            }
            
            .chat-messages {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .message {
                padding: 15px 20px;
                border-radius: 18px;
                max-width: 80%;
                word-wrap: break-word;
                animation: fadeIn 0.3s ease-in;
                font-weight: 400;
                letter-spacing: 0.3px;
                line-height: 1.6;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message.user {
                align-self: flex-end;
                background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            
            .message.bot {
                align-self: flex-start;
                background: #2a2a2a;
                color: #f0f0f0;
                border: 1px solid #e50914;
                border-bottom-left-radius: 5px;
            }
            
            .message.loading {
                align-self: flex-start;
                background: #333;
                color: #e50914;
                font-style: italic;
            }
            
            .search-container {
                padding: 20px 30px;
                background: #1a1a1a;
                border-top: 2px solid #e50914;
            }
            
            .search-box {
                display: flex;
                gap: 10px;
            }
            
            #userInput {
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #333;
                background: #0a0a0a;
                color: #f0f0f0;
                border-radius: 12px;
                font-size: 16px;
                outline: none;
                transition: all 0.3s ease;
                font-family: 'Poppins', sans-serif;
                font-weight: 500;
                letter-spacing: 0.5px;
            }
            
            #userInput::placeholder {
                color: #666;
            }
            
            #userInput:focus {
                border-color: #e50914;
                box-shadow: 0 0 0 3px rgba(229, 9, 20, 0.2);
            }
            
            .btn-search {
                padding: 15px 30px;
                background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
                font-family: 'Poppins', sans-serif;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            
            .btn-search:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6);
            }
            
            .btn-search:active {
                transform: translateY(0);
            }
            
            .result-item {
                background: #2a2a2a;
                border: 1px solid #e50914;
                border-radius: 12px;
                padding: 15px;
                margin-top: 10px;
                transition: all 0.3s ease;
                animation: slideInUp 0.5s ease-out;
                position: relative;
                overflow: hidden;
            }
            
            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .result-item:hover {
                border-color: #e50914;
                box-shadow: 0 8px 20px rgba(229, 9, 20, 0.4);
                transform: translateY(-2px);
            }
            
            .result-item::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(229, 9, 20, 0.2), transparent);
                transition: left 0.5s ease;
            }
            
            .result-item:hover::before {
                left: 100%;
            }
            
            .result-title {
                font-weight: 600;
                color: #e50914;
                font-size: 1.15em;
                margin-bottom: 5px;
                animation: pulse 2s ease-in-out infinite;
                letter-spacing: 0.3px;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }
            
            .result-info {
                color: #aaa;
                font-size: 0.95em;
                margin-bottom: 5px;
                letter-spacing: 0.2px;
            }
            
            .result-description {
                color: #d0d0d0;
                font-size: 1em;
                margin-top: 10px;
                padding: 10px;
                background: #1a1a1a;
                border-radius: 8px;
                border-left: 3px solid #e50914;
                line-height: 1.6;
                letter-spacing: 0.2px;
            }
            
            .result-description strong {
                color: #e50914;
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
            }
            
            .error-message {
                background: #2a0000;
                color: #ff4444;
                padding: 15px;
                border-radius: 10px;
                margin-top: 10px;
                border: 1px solid #e50914;
            }
            
            .suggestion-message {
                background: linear-gradient(135deg, #1a0a0a 0%, #2a1a1a 100%);
                color: #ff6666;
                border: 2px solid #e50914;
            }
            
            .cursor {
                animation: blink 1s infinite;
            }
            
            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }
            
            .header h1 {
                animation: bounce 1s ease-in-out;
            }
            
            @keyframes bounce {
                0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                40% { transform: translateY(-10px); }
                60% { transform: translateY(-5px); }
            }
            
            /* Estilos para películas sugeridas tipo streaming */
            .featured-movies {
                padding: 20px 30px;
                background: #1a1a1a;
                border-bottom: 2px solid #e50914;
            }
            
            .featured-title {
                color: #e50914;
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .movies-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 15px;
                overflow-x: auto;
                padding-bottom: 10px;
            }
            
            .movie-card {
                background: #2a2a2a;
                border-radius: 12px;
                overflow: hidden;
                cursor: pointer;
                transition: all 0.3s ease;
                border: 2px solid transparent;
                position: relative;
            }
            
            .movie-card:hover {
                transform: scale(1.05) translateY(-5px);
                border-color: #e50914;
                box-shadow: 0 10px 30px rgba(229, 9, 20, 0.4);
            }
            
            .movie-poster {
                width: 100%;
                height: 200px;
                background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 3em;
                color: white;
                position: relative;
            }
            
            .movie-info {
                padding: 10px;
                background: #2a2a2a;
            }
            
            .movie-card-title {
                color: #fff;
                font-size: 0.9em;
                font-weight: 600;
                margin-bottom: 5px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .movie-card-type {
                color: #aaa;
                font-size: 0.8em;
            }
            
            .play-icon {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                opacity: 0;
                transition: all 0.3s ease;
                background: rgba(229, 9, 20, 0.9);
                border-radius: 50%;
                width: 50px;
                height: 50px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5em;
            }
            
            .movie-card:hover .play-icon {
                opacity: 1;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎬 CINEBOT</h1>
                <p>Tu asistente personal de recomendaciones de películas</p>
            </div>
            
            <div class="featured-movies" id="featuredMovies">
                <div class="featured-title">🎥 En Tendencia</div>
                <div class="movies-grid" id="moviesGrid">
                    <!-- Las películas se cargarán aquí con JavaScript -->
                </div>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot">
                        👋 ¡Hola, qué quieres ver hoy!
                        <br><br>
                        <small>💡 Ejemplos: "acción thriller", "comedia romántica", "drama histórico"</small>
                    </div>
                </div>
            </div>
            
            <div class="search-container">
                <div class="search-box">
                    <input 
                        type="text" 
                        id="userInput" 
                        placeholder="Escribe tu búsqueda... (ej: thriller de acción)"
                        onkeypress="if(event.key === 'Enter') searchMovies()"
                    >
                    <button class="btn-search" onclick="searchMovies()">
                        🔍 Buscar
                    </button>
                </div>
            </div>
        </div>
        
        <script>
            // Almacenar búsquedas recientes
            let recentSearches = JSON.parse(localStorage.getItem('recentSearches')) || [];
            
            // Función para cargar películas destacadas
            async function loadFeaturedMovies() {
                try {
                    const response = await fetch('/peliculas');
                    const data = await response.json();
                    
                    if (data.peliculas && data.peliculas.length > 0) {
                        // Seleccionar películas aleatorias
                        const randomMovies = data.peliculas.sort(() => 0.5 - Math.random()).slice(0, 8);
                        
                        const moviesGrid = document.getElementById('moviesGrid');
                        moviesGrid.innerHTML = '';
                        
                        randomMovies.forEach((movie, index) => {
                            const movieCard = document.createElement('div');
                            movieCard.className = 'movie-card';
                            movieCard.innerHTML = `
                                <div class="movie-poster">
                                    ${movie.title ? movie.title.charAt(0) : '🎬'}
                                    <div class="play-icon">▶</div>
                                </div>
                                <div class="movie-info">
                                    <div class="movie-card-title">${movie.title || 'Sin título'}</div>
                                    <div class="movie-card-type">${movie.type || 'N/A'}</div>
                                </div>
                            `;
                            
                            // Agregar evento de clic
                            movieCard.addEventListener('click', () => {
                                document.getElementById('userInput').value = movie.description || movie.title || '';
                                showMovieDetails(movie);
                            });
                            
                            moviesGrid.appendChild(movieCard);
                        });
                    }
                } catch (error) {
                    console.error('Error al cargar películas:', error);
                }
            }
            
            // Función para mostrar detalles de una película
            function showMovieDetails(movie) {
                const chatMessages = document.getElementById('chatMessages');
                const message = document.createElement('div');
                message.className = 'message bot';
                message.innerHTML = `
                    <strong>🎬 ${movie.title || 'Sin título'}</strong><br>
                    📺 Tipo: ${movie.type || 'N/A'} | 📅 Año: ${movie.release_year || 'N/A'}<br><br>
                    <strong>📝 Descripción:</strong><br>
                    ${movie.description || 'Sin descripción disponible'}
                `;
                chatMessages.appendChild(message);
            }
            
            // Animación de escritura para el saludo inicial
            window.addEventListener('DOMContentLoaded', function() {
                loadFeaturedMovies();
                const welcomeMessage = document.querySelector('.message.bot');
                if (welcomeMessage) {
                    const originalText = welcomeMessage.innerHTML;
                    welcomeMessage.innerHTML = '';
                    let charIndex = 0;
                    const text = originalText.replace(/<[^>]*>/g, '');
                    
                    function typeWriter() {
                        if (charIndex < text.length) {
                            welcomeMessage.innerHTML = originalText.substring(0, charIndex) + '<span class="cursor">|</span>';
                            charIndex++;
                            setTimeout(typeWriter, 30);
                        } else {
                            welcomeMessage.innerHTML = originalText;
                        }
                    }
                    
                    setTimeout(typeWriter, 500);
                }
            });
            
            async function searchMovies() {
                const input = document.getElementById('userInput');
                const query = input.value.trim();
                
                if (!query || query.length < 3) {
                    alert('Por favor, ingresa al menos 3 caracteres para buscar');
                    return;
                }
                
                // Agregar mensaje del usuario
                addMessage(query, 'user');
                input.value = '';
                
                // Guardar búsqueda reciente
                if (!recentSearches.includes(query.toLowerCase())) {
                    recentSearches.unshift(query);
                    recentSearches = recentSearches.slice(0, 5); // Mantener solo las últimas 5
                    localStorage.setItem('recentSearches', JSON.stringify(recentSearches));
                }
                
                // Mostrar mensaje de carga
                const loadingId = addMessage('🔍 Buscando películas...', 'loading');
                
                try {
                    const response = await fetch(`/peliculas/buscar/${encodeURIComponent(query)}`);
                    const data = await response.json();
                    
                    // Remover mensaje de carga
                    removeMessage(loadingId);
                    
                    if (!response.ok) {
                        addMessage(`❌ ${data.detail}`, 'bot');
                        // Preguntar qué más quiere
                        setTimeout(() => {
                            addMessage('💬 ¿Qué otra película te gustaría buscar?', 'bot');
                        }, 500);
                        return;
                    }
                    
                    // Mostrar resultados
                    if (data.peliculas && data.peliculas.length > 0) {
                        let botMessage = `✅ Encontré ${data.total_encontrado} película(s) relacionada(s) con "<strong>${data.descripcion_buscada}</strong>". Te muestro las ${data.total_mostrado} más relevantes:\n\n`;
                        
                        data.peliculas.forEach((pelicula, index) => {
                            const title = pelicula.title || 'Sin título';
                            const type = pelicula.type || 'N/A';
                            const year = pelicula.release_year || 'N/A';
                            const descripcion = pelicula.description || 'Sin descripción disponible';
                            const similarity = pelicula.similitud || 0;
                            
                            botMessage += `<div class="result-item">
                                <div class="result-title">🎬 ${index + 1}. ${title}</div>
                                <div class="result-info">
                                    📺 Tipo: <strong>${type}</strong> | 
                                    📅 Año: <strong>${year}</strong> | 
                                    🎯 Similitud: <strong>${similarity}%</strong>
                                </div>
                                <div class="result-description">
                                    <strong>📝 Descripción:</strong><br>
                                    ${descripcion}
                                </div>
                            </div>`;
                        });
                        
                        addMessage(botMessage, 'bot');
                        
                        // Preguntar qué más quiere buscar con sugerencias
                        setTimeout(() => {
                            showSuggestions();
                        }, 1000);
                    } else {
                        addMessage('No se encontraron películas relacionadas con tu búsqueda.', 'bot');
                        setTimeout(() => {
                            addMessage('💬 Intenta con otros términos. ¿Qué otra búsqueda quieres hacer?', 'bot');
                        }, 500);
                    }
                } catch (error) {
                    removeMessage(loadingId);
                    addMessage('❌ Error al buscar películas. Por favor, intenta nuevamente.', 'bot');
                    console.error('Error:', error);
                    setTimeout(() => {
                        addMessage('💬 ¿Qué otra búsqueda quieres realizar?', 'bot');
                    }, 500);
                }
            }
            
            function addMessage(text, type) {
                const messagesDiv = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                
                // Detectar si es un mensaje de sugerencia
                const isSuggestion = text.includes('¿Qué otra') || text.includes('¿Qué otra búsqueda');
                const className = isSuggestion ? `message ${type} suggestion-message` : `message ${type}`;
                
                messageDiv.className = className;
                messageDiv.innerHTML = text;
                messageDiv.id = 'msg-' + Date.now();
                messagesDiv.appendChild(messageDiv);
                
                // Auto-scroll con suavidad
                const container = document.getElementById('chatContainer');
                container.scrollTo({
                    top: container.scrollHeight,
                    behavior: 'smooth'
                });
                
                return messageDiv.id;
            }
            
            function removeMessage(id) {
                const msg = document.getElementById(id);
                if (msg) msg.remove();
            }
            
            function showSuggestions() {
                let message = '💬 ¿Qué otra sugerencia quieres? ';
                
                if (recentSearches.length > 0) {
                    message += '<br><br><strong>🕒 Búsquedas recientes:</strong><br>';
                    recentSearches.slice(0, 3).forEach((search, index) => {
                        message += `<span style="background: #e50914; color: white; padding: 5px 10px; border-radius: 5px; margin: 3px; display: inline-block; cursor: pointer; font-size: 0.9em;" onclick="document.getElementById('userInput').value='${search}'; searchMovies();">${search}</span>`;
                    });
                } else {
                    message += 'Describe otra película que te gustaría ver.';
                }
                
                addMessage(message, 'bot');
            }
        </script>
    </body>
    </html>
    """

# LISTA DE PELÍCULAS: Obtener todas las películas del dataset
@app.get("/peliculas")
async def lista_peliculas():
    """
    Ruta para obtener la lista completa de todas las películas disponibles en el dataset
    """
    global dataset_netflix
    
    if dataset_netflix is None:
        raise HTTPException(status_code=500, detail="El dataset no está disponible")
    
    try:
        # Convertir el DataFrame a diccionario para JSON
        peliculas = dataset_netflix.to_dict(orient='records')
        
        return JSONResponse(
            content={
                "total": len(peliculas),
                "peliculas": peliculas
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar los datos: {str(e)}")

# FILTRO POR ID: Obtener una película específica por su ID
@app.get("/pelicula/id/{show_id}")
async def pelicula_por_id(show_id: str):
    """
    Ruta para obtener una película específica según su ID
    
    Args:
        show_id: ID de la película (ej: 's1', 's2', etc.)
    """
    global dataset_netflix
    
    if dataset_netflix is None:
        raise HTTPException(status_code=500, detail="El dataset no está disponible")
    
    try:
        # Buscar la película por ID
        pelicula = dataset_netflix[dataset_netflix['show_id'] == show_id]
        
        if pelicula.empty:
            raise HTTPException(status_code=404, detail=f"Película con ID '{show_id}' no encontrada")
        
        # Convertir a diccionario
        pelicula_dict = pelicula.to_dict(orient='records')[0]
        
        return JSONResponse(content=pelicula_dict)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al buscar la película: {str(e)}")

# FILTRO POR CATEGORÍA/GÉNERO: Obtener películas según categoría
@app.get("/peliculas/categoria/{categoria}")
async def peliculas_por_categoria(categoria: str):
    """
    Ruta para obtener lista de películas según la categoría/género solicitado por el usuario
    
    Args:
        categoria: Categoría o género a buscar (ej: 'Dramas', 'Comedies', 'Action & Adventure', etc.)
    """
    global dataset_netflix
    
    if dataset_netflix is None:
        raise HTTPException(status_code=500, detail="El dataset no está disponible")
    
    try:
        # Buscar películas que contengan la categoría solicitada
        # La columna 'listed_in' puede contener múltiples categorías separadas por coma
        peliculas = dataset_netflix[
            dataset_netflix['listed_in'].str.contains(categoria, case=False, na=False)
        ]
        
        if peliculas.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron películas para la categoría '{categoria}'"
            )
        
        # Convertir a lista de diccionarios
        peliculas_list = peliculas.to_dict(orient='records')
        
        return JSONResponse(
            content={
                "categoria": categoria,
                "total": len(peliculas_list),
                "peliculas": peliculas_list
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al buscar películas: {str(e)}")

# ETAPA 7: Ruta del chatbot - Filtro por descripción

def procesar_texto(texto):
    """
    Procesa el texto del usuario para extraer palabras clave
    """
    if not texto or pd.isna(texto):
        return []
    
    # Convertir a minúsculas y tokenizar
    tokens = word_tokenize(str(texto).lower())
    
    # Filtrar tokens (solo palabras, sin signos de puntuación)
    tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
    
    return tokens

def calcular_similitud(descripcion_usuario, descripcion_pelicula):
    """
    Calcula la similitud entre la descripción del usuario y la descripción de una película
    usando procesamiento de texto con NLTK
    """
    if pd.isna(descripcion_usuario) or pd.isna(descripcion_pelicula):
        return 0.0
    
    # Obtener palabras clave de ambas descripciones
    palabras_usuario = set(procesar_texto(descripcion_usuario))
    palabras_pelicula = set(procesar_texto(descripcion_pelicula))
    
    if not palabras_usuario or not palabras_pelicula:
        return 0.0
    
    # Calcular similitud usando intersección de palabras
    palabras_comunes = palabras_usuario.intersection(palabras_pelicula)
    similitud = len(palabras_comunes) / max(len(palabras_usuario), len(palabras_pelicula))
    
    return similitud

@app.get("/peliculas/buscar/{descripcion}")
async def buscar_peliculas_por_descripcion(descripcion: str):
    """
    Ruta del chatbot para obtener lista de películas que concuerden con la descripción del usuario
    Usa procesamiento de lenguaje natural para encontrar películas similares
    
    Args:
        descripcion: Descripción del usuario sobre lo que busca (ej: "acción thriller", "comedia romántica", etc.)
    """
    global dataset_netflix
    
    if dataset_netflix is None:
        raise HTTPException(status_code=500, detail="El dataset no está disponible")
    
    if not descripcion or len(descripcion.strip()) < 3:
        raise HTTPException(
            status_code=400, 
            detail="La descripción debe tener al menos 3 caracteres"
        )
    
    try:
        # Buscar películas con similitud en la descripción
        resultados = []
        
        for _, pelicula in dataset_netflix.iterrows():
            if pd.notna(pelicula.get('description')):
                similitud = calcular_similitud(descripcion, pelicula['description'])
                
                if similitud > 0:
                    pelicula_dict = pelicula.to_dict()
                    pelicula_dict['similitud'] = round(similitud * 100, 2)
                    resultados.append(pelicula_dict)
        
        # Ordenar por similitud (de mayor a menor)
        resultados = sorted(resultados, key=lambda x: x['similitud'], reverse=True)
        
        if not resultados:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron películas relacionadas con: '{descripcion}'"
            )
        
        # Tomar solo las top 20 más relevantes
        top_resultados = resultados[:20]
        
        return JSONResponse(
            content={
                "descripcion_buscada": descripcion,
                "total_encontrado": len(resultados),
                "total_mostrado": len(top_resultados),
                "peliculas": top_resultados
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al buscar películas: {str(e)}")

# Bloque para ejecutar el servidor
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Iniciando servidor de la API...")
    print("📍 Accede a: http://localhost:8000")
    print("📚 Documentación: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)

