import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Database, Search, Zap, Brain, Calculator, BarChart3, FileText, Music, ShoppingCart, MessageCircle, MapPin, Coffee, BookOpen, Navigation, Grid3x3, Layers, Target } from 'lucide-react';

const VectorDBPresentation = () => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [demoQuery, setDemoQuery] = useState([2, 8, 7]);
  const [selectedMetric, setSelectedMetric] = useState('both');
  const [showBeforeAfter, setShowBeforeAfter] = useState('before');
  const [show3D, setShow3D] = useState(false);

  const movies = [
    { name: "La La Land", vector: [2, 8, 7], genre: "Romance/Musical", description: "Romantic musical about dreams" },
    { name: "Mad Max", vector: [9, 2, 3], genre: "Action", description: "Post-apocalyptic action thriller" },
    { name: "The Hangover", vector: [3, 4, 9], genre: "Comedy", description: "Comedy about bachelor party gone wrong" },
    { name: "Titanic", vector: [4, 9, 2], genre: "Romance/Drama", description: "Epic romance on doomed ship" },
    { name: "John Wick", vector: [10, 1, 2], genre: "Action", description: "Stylized action revenge story" },
    { name: "When Harry Met Sally", vector: [1, 9, 6], genre: "Romance/Comedy", description: "Friends to lovers romantic comedy" }
  ];

  const locations = [
    { name: "Starbucks Landmark 81", vector: [8, 3, 9], type: "Coffee Shop" },
    { name: "Highlands Coffee", vector: [7, 4, 8], type: "Coffee Shop" },
    { name: "Pizza 4P's", vector: [5, 8, 6], type: "Restaurant" },
    { name: "Lotte Mart", vector: [3, 9, 4], type: "Shopping" },
    { name: "Vincom Center", vector: [4, 9, 5], type: "Shopping" }
  ];

  const calculateSimilarity = (query, item) => {
    const euclidean = Math.sqrt(
      query.reduce((sum, val, i) => sum + Math.pow(val - item.vector[i], 2), 0)
    );
    
    const dotProduct = query.reduce((sum, val, i) => sum + val * item.vector[i], 0);
    const queryMagnitude = Math.sqrt(query.reduce((sum, val) => sum + val * val, 0));
    const itemMagnitude = Math.sqrt(item.vector.reduce((sum, val) => sum + val * val, 0));
    const cosine = dotProduct / (queryMagnitude * itemMagnitude);
    
    return { euclidean, cosine };
  };

  const getSortedResults = () => {
    return movies.map(movie => ({
      ...movie,
      ...calculateSimilarity(demoQuery, movie)
    })).sort((a, b) => {
      if (selectedMetric === 'euclidean') return a.euclidean - b.euclidean;
      if (selectedMetric === 'cosine') return b.cosine - a.cosine;
      return a.euclidean - b.euclidean;
    });
  };

  const beforeAfterExamples = {
    before: [
      { query: '"comfortable shoes"', results: ['No results found'] },
      { query: '"cozy coffee"', results: ['No results found'] },
      { query: '"fast breakfast"', results: ['No results found'] }
    ],
    after: [
      { query: '"comfortable shoes"', results: ['Comfy Sneakers', 'Soft Athletic Shoes', 'Cushioned Runners'] },
      { query: '"cozy coffee"', results: ['Warm Café', 'Intimate Coffee House', 'Cosy Espresso Bar'] },
      { query: '"fast breakfast"', results: ['Quick Morning Meals', 'Express Breakfast', 'Rapid AM Options'] }
    ]
  };

  const nextSlide = () => setCurrentSlide((prev) => (prev + 1) % slides.length);
  const prevSlide = () => setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);

  const slides = [
    // Slide 1: Hook
    {
      title: "Câu hỏi thử thách",
      content: (
        <div className="text-center py-16">
          <div className="mb-12">
            <h1 className="text-5xl font-bold text-gray-800 mb-8">
              Tại sao database nhanh nhất cho use case của bạn...
            </h1>
            <h2 className="text-4xl font-bold text-blue-600 mb-8">
              có thể không phải là database?
            </h2>
          </div>
          
          <div className="bg-gradient-to-r from-red-50 to-orange-50 p-8 rounded-lg mb-8">
            <p className="text-xl text-gray-700 mb-4">
              <strong>60% queries trả về "No results"</strong> dù product có sẵn
            </p>
            <p className="text-lg text-gray-600">
              Lý do thực sự không phải do algorithm...
            </p>
          </div>

          <div className="text-lg text-gray-700">
            <p>Hôm nay chúng ta sẽ khám phá:</p>
            <p className="mt-2 font-semibold">Vector Database - từ khái niệm cơ bản đến ứng dụng thực tế</p>
          </div>
        </div>
      )
    },

    // Slide 2: Problem Definition
    {
      title: "Vấn đề cốt lõi: Exact Match vs Semantic Similarity",
      content: (
        <div className="space-y-8">
          <div className="bg-red-50 p-6 rounded-lg border-l-4 border-red-400">
            <h3 className="text-xl font-bold text-red-700 mb-4">📱 Traditional Database: Tìm kiếm theo từ khóa chính xác</h3>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h4 className="font-bold mb-2">Customer search:</h4>
                <ul className="text-gray-700 space-y-1">
                  <li>• "comfortable sneakers" → 0 results</li>
                  <li>• "cozy coffee shop" → 0 results</li>
                  <li>• "fast breakfast ideas" → 0 results</li>
                </ul>
              </div>
              <div>
                <h4 className="font-bold mb-2">Products in MySQL:</h4>
                <ul className="text-gray-700 space-y-1">
                  <li>• "comfy athletic shoes" ✅</li>
                  <li>• "warm café atmosphere" ✅</li>
                  <li>• "quick morning meals" ✅</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="bg-gray-50 p-6 rounded-lg">
              <h4 className="font-bold text-gray-700 mb-4 flex items-center">
                <Database className="w-5 h-5 mr-2" />
                Traditional DBMS
              </h4>
              <ul className="space-y-2 text-gray-700">
                <li>• Tìm kiếm <strong>exact match</strong></li>
                <li>• SQL: WHERE title = "La La Land"</li>
                <li>• Indexing theo B-Tree, Hash</li>
                <li>• Nhanh nhưng cứng nhắc</li>
                <li>• "iPhone 13" ≠ "iPhone mới"</li>
              </ul>
            </div>
            
            <div className="bg-green-50 p-6 rounded-lg">
              <h4 className="font-bold text-green-700 mb-4 flex items-center">
                <Search className="w-5 h-5 mr-2" />
                Vector Database
              </h4>
              <ul className="space-y-2 text-gray-700">
                <li>• Tìm kiếm <strong>semantic similarity</strong></li>
                <li>• Vector similarity: cosine(A, B)</li>
                <li>• ANN indexing (HNSW, IVF)</li>
                <li>• Thông minh, hiểu ngữ cảnh</li>
                <li>• "iPhone 13" ≈ "điện thoại Apple mới"</li>
              </ul>
            </div>
          </div>

          <div className="bg-blue-50 p-6 rounded-lg">
            <p className="text-lg text-blue-700 font-semibold">
              💡 Vấn đề cốt lõi: Database thiết kế cho "exact match" vs User thinking "semantic similarity"
            </p>
          </div>
        </div>
      )
    },

    // Slide 3: Vector Database Definition
    {
      title: "Vector Database: Định nghĩa & Khái niệm cơ bản",
      content: (
        <div className="space-y-8">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-center mb-6">🗃️ Vector Database = Database lưu trữ & tìm kiếm vectors</h3>
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="space-y-4">
              <h4 className="font-bold text-gray-800 mb-4">📊 Vector là gì?</h4>
              <div className="bg-white p-4 rounded-lg border">
                <p className="text-sm text-gray-600 mb-2">Một list số thực biểu diễn ý nghĩa:</p>
                <div className="font-mono text-sm bg-gray-100 p-2 rounded">
                  "La La Land" → [2.1, 8.7, 6.9, ...]
                </div>
                <p className="text-xs text-gray-500 mt-1">384-3072 dimensions</p>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg">
                <h5 className="font-bold text-yellow-700 mb-2">Embedding Process:</h5>
                <div className="text-sm space-y-1">
                  <p>Text/Image/Audio → AI Model → Vector</p>
                  <p className="text-xs text-gray-600">OpenAI, Cohere, Sentence-BERT...</p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-bold text-gray-800 mb-4">🎯 Tại sao dùng Vector?</h4>
              <div className="space-y-3">
                <div className="bg-green-50 p-3 rounded">
                  <p className="font-semibold text-green-700">Semantic Understanding</p>
                  <p className="text-sm text-gray-600">Vectors gần nhau = ý nghĩa giống nhau</p>
                </div>
                <div className="bg-blue-50 p-3 rounded">
                  <p className="font-semibold text-blue-700">Mathematical Operations</p>
                  <p className="text-sm text-gray-600">Cosine similarity, Vector arithmetic</p>
                </div>
                <div className="bg-purple-50 p-3 rounded">
                  <p className="font-semibold text-purple-700">Scalability</p>
                  <p className="text-sm text-gray-600">ANN algorithms cho millions vectors</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg">
            <h4 className="font-bold text-gray-800 mb-4">🌍 Thư viện địa lý của ý nghĩa</h4>
            <p className="text-gray-700 mb-4">
              Hãy tưởng tượng sắp xếp thư viện không theo alphabet, mà theo <strong>ý nghĩa</strong>:
            </p>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <p>📖 Sách về chó</p>
                <p className="text-gray-500">↓ gần nhau</p>
                <p>🐕 Sách về thú cưng</p>
                <p className="text-gray-500">↓ gần nhau</p>
                <p>🦁 Sách về động vật</p>
              </div>
              <div className="text-center">
                <p>🎬 Phim hành động</p>
                <p className="text-gray-500">↓ gần nhau</p>
                <p>🚗 Phim tốc độ</p>
                <p className="text-gray-500">↓ gần nhau</p>
                <p>💥 Phim thriller</p>
              </div>
              <div className="text-center">
                <p>☕ Coffee shop</p>
                <p className="text-gray-500">↓ gần nhau</p>
                <p>🥐 Café & bakery</p>
                <p className="text-gray-500">↓ gần nhau</p>
                <p>🍰 Tea house</p>
              </div>
            </div>
          </div>
        </div>
      )
    },

    // Slide 4: Vector Visualization 2D/3D
    {
      title: "Hiểu Vector qua hình ảnh 2D & 3D",
      content: (
        <div className="space-y-6">
          <div className="flex justify-center mb-4">
            <div className="bg-white rounded-lg p-2 border">
              <button 
                onClick={() => setShow3D(false)}
                className={`px-4 py-2 rounded mr-2 ${!show3D ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
              >
                2D View
              </button>
              <button 
                onClick={() => setShow3D(true)}
                className={`px-4 py-2 rounded ${show3D ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
              >
                3D View
              </button>
            </div>
          </div>

          {!show3D ? (
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg border">
                <h4 className="font-bold mb-4">📊 2D Vector Space: [Romance, Action]</h4>
                <div className="relative bg-gray-50 h-64 rounded border">
                  <div className="absolute bottom-0 left-0 w-full h-px bg-gray-400"></div>
                  <div className="absolute bottom-0 left-0 w-px h-full bg-gray-400"></div>
                  
                  <div className="absolute bottom-1 right-2 text-xs text-gray-600">Action →</div>
                  <div className="absolute top-2 left-1 text-xs text-gray-600 transform -rotate-90 origin-left">Romance ↑</div>
                  
                  <div className="absolute bottom-4 left-8 w-2 h-2 bg-red-500 rounded-full" title="La La Land [2,8]"></div>
                  <div className="absolute bottom-12 right-4 w-2 h-2 bg-blue-500 rounded-full" title="Mad Max [9,2]"></div>
                  <div className="absolute bottom-6 left-16 w-2 h-2 bg-green-500 rounded-full" title="Titanic [4,9]"></div>
                  <div className="absolute bottom-16 right-2 w-2 h-2 bg-purple-500 rounded-full" title="John Wick [10,1]"></div>
                  
                  <div className="absolute bottom-8 left-12 w-3 h-3 bg-yellow-500 rounded-full border-2 border-yellow-700" title="Query [2,8]"></div>
                </div>
                <div className="mt-2 text-xs space-y-1">
                  <div><span className="inline-block w-2 h-2 bg-red-500 rounded mr-1"></span>La La Land [2,8]</div>
                  <div><span className="inline-block w-2 h-2 bg-green-500 rounded mr-1"></span>Titanic [4,9]</div>
                  <div><span className="inline-block w-2 h-2 bg-yellow-500 rounded mr-1"></span>Query [2,8] - Gần La La Land nhất!</div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h5 className="font-bold text-blue-700 mb-2">💡 Quan sát quan trọng:</h5>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>• Points gần nhau = ý nghĩa giống nhau</li>
                    <li>• La La Land & Titanic (romance cao) cluster lại</li>
                    <li>• Mad Max & John Wick (action cao) ở góc khác</li>
                    <li>• Distance = Semantic difference</li>
                  </ul>
                </div>

                <div className="bg-yellow-50 p-4 rounded-lg">
                  <h5 className="font-bold text-yellow-700 mb-2">🎯 Similarity Search:</h5>
                  <p className="text-sm text-gray-700">
                    Query [2,8] = "romantic movie with little action"
                    <br />→ Gần nhất: La La Land (cùng có romance cao)
                  </p>
                </div>

                <div className="bg-purple-50 p-4 rounded-lg">
                  <h5 className="font-bold text-purple-700 mb-2">📏 Distance Calculation:</h5>
                  <div className="text-xs font-mono space-y-1">
                    <p>Euclidean: √[(2-2)² + (8-8)²] = 0</p>
                    <p>Mad Max: √[(2-9)² + (8-2)²] = 9.22</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg border">
                <h4 className="font-bold mb-4">🎭 3D Vector Space: [Action, Romance, Comedy]</h4>
                <div className="relative bg-gray-50 h-64 rounded border">
                  <div className="absolute inset-4">
                    <div className="absolute top-4 left-4 w-32 h-32 border border-gray-300 transform rotate-12"></div>
                    <div className="absolute top-8 left-8 w-32 h-32 border-2 border-gray-600"></div>
                    
                    <div className="absolute top-12 left-12 w-2 h-2 bg-red-500 rounded-full shadow-lg" title="La La Land [2,8,7]"></div>
                    <div className="absolute top-20 left-32 w-2 h-2 bg-blue-500 rounded-full shadow-lg" title="Mad Max [9,2,3]"></div>
                    <div className="absolute top-8 left-20 w-2 h-2 bg-green-500 rounded-full shadow-lg" title="Hangover [3,4,9]"></div>
                    <div className="absolute top-16 left-16 w-3 h-3 bg-yellow-500 rounded-full border-2 border-yellow-700" title="Query [2,8,7]"></div>
                    
                    <div className="absolute bottom-2 right-2 text-xs text-gray-600">Action</div>
                    <div className="absolute top-2 left-2 text-xs text-gray-600">Romance</div>
                    <div className="absolute top-2 right-2 text-xs text-gray-600">Comedy</div>
                  </div>
                </div>
                <p className="text-xs text-gray-600 mt-2">3D space cho phép biểu diễn nhiều thuộc tính hơn</p>
              </div>

              <div className="space-y-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <h5 className="font-bold text-green-700 mb-2">🚀 Scaling to High Dimensions:</h5>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li>• Real embeddings: 384-3072 dimensions</li>
                    <li>• Mỗi dimension = 1 aspect của meaning</li>
                    <li>• Không visualize được, nhưng math vẫn work</li>
                    <li>• Cosine similarity scales tốt</li>
                  </ul>
                </div>

                <div className="bg-orange-50 p-4 rounded-lg">
                  <h5 className="font-bold text-orange-700 mb-2">🎬 Movie Vector Example:</h5>
                  <div className="text-xs space-y-1">
                    <p><strong>La La Land [2,8,7]:</strong></p>
                    <p>• Action: 2/10 (ít hành động)</p>
                    <p>• Romance: 8/10 (rất lãng mạn)</p>
                    <p>• Comedy: 7/10 (khá vui nhộn)</p>
                  </div>
                </div>

                <div className="bg-blue-50 p-4 rounded-lg">
                  <h5 className="font-bold text-blue-700 mb-2">📍 Vector Geography:</h5>
                  <p className="text-sm text-gray-700">
                    Trong không gian 3D, romance movies cluster ở vùng có Romance coordinate cao, 
                    action movies ở vùng Action cao.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )
    },

    // Slide 5: Math Formulas
    {
      title: "Công thức tính Similarity: Euclidean vs Cosine",
      content: (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg text-center">
            <h3 className="text-2xl font-bold text-gray-800 mb-4">🔢 Toán học đằng sau Vector Similarity</h3>
            <p className="text-lg text-gray-600">Hai cách chính để đo độ "gần" giữa hai vectors</p>
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="bg-white p-6 rounded-lg border">
              <h4 className="font-bold text-blue-700 mb-4 flex items-center">
                <Calculator className="w-5 h-5 mr-2" />
                Euclidean Distance
              </h4>
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded">
                  <h5 className="font-bold mb-2">Công thức:</h5>
                  <div className="font-mono text-lg text-center bg-white p-3 rounded border">
                    d(A,B) = √[Σ(aᵢ - bᵢ)²]
                  </div>
                </div>
                
                <div className="bg-gray-50 p-4 rounded">
                  <h5 className="font-bold mb-2">Ví dụ thực tế:</h5>
                  <div className="text-sm space-y-1">
                    <p>A = [2, 8, 7] (La La Land)</p>
                    <p>B = [4, 9, 2] (Titanic)</p>
                    <p className="font-mono">d = √[(2-4)² + (8-9)² + (7-2)²]</p>
                    <p className="font-mono">d = √[4 + 1 + 25] = √30 ≈ 5.48</p>
                  </div>
                </div>

                <div className="bg-yellow-50 p-3 rounded">
                  <p className="text-sm"><strong>Đặc điểm:</strong> Càng nhỏ càng gần. Affected by magnitude.</p>
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-lg border">
              <h4 className="font-bold text-green-700 mb-4 flex items-center">
                <Target className="w-5 h-5 mr-2" />
                Cosine Similarity
              </h4>
              <div className="space-y-4">
                <div className="bg-green-50 p-4 rounded">
                  <h5 className="font-bold mb-2">Công thức:</h5>
                  <div className="font-mono text-lg text-center bg-white p-3 rounded border">
                    cos(A,B) = A·B / (||A|| × ||B||)
                  </div>
                </div>
                
                <div className="bg-gray-50 p-4 rounded">
                  <h5 className="font-bold mb-2">Ví dụ thực tế:</h5>
                  <div className="text-sm space-y-1">
                    <p>A = [2, 8, 7], B = [4, 9, 2]</p>
                    <p className="font-mono">A·B = 2×4 + 8×9 + 7×2 = 94</p>
                    <p className="font-mono">||A|| = √(4+64+49) = 10.82</p>
                    <p className="font-mono">||B|| = √(16+81+4) = 10.05</p>
                    <p className="font-mono">cos = 94/(10.82×10.05) ≈ 0.865</p>
                  </div>
                </div>

                <div className="bg-yellow-50 p-3 rounded">
                  <p className="text-sm"><strong>Đặc điểm:</strong> Gần 1 = giống nhau. Measures direction, not magnitude.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 p-6 rounded-lg">
            <h4 className="font-bold text-orange-700 mb-4">🎯 Khi nào dùng gì?</h4>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h5 className="font-bold text-blue-600 mb-2">Euclidean Distance khi:</h5>
                <ul className="text-sm space-y-1">
                  <li>• Magnitude (độ lớn) quan trọng</li>
                  <li>• Coordinates có cùng đơn vị</li>
                  <li>• Geometric space problems</li>
                  <li>• Ví dụ: GPS coordinates, image pixels</li>
                </ul>
              </div>
              <div>
                <h5 className="font-bold text-green-600 mb-2">Cosine Similarity khi:</h5>
                <ul className="text-sm space-y-1">
                  <li>• Direction (hướng) quan trọng hơn magnitude</li>
                  <li>• Text embeddings, recommendations</li>
                  <li>• Document length không quan trọng</li>
                  <li>• Ví dụ: Search relevance, content similarity</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg border-l-4 border-purple-400">
            <h4 className="font-bold text-purple-700 mb-2">🔦 Intuition: Flashlight Analogy</h4>
            <p className="text-gray-700">
              Tưởng tượng hai chiếc đèn pin chiếu từ cùng một điểm. Cosine similarity đo góc giữa chúng:
              <br />• Cùng hướng (góc nhỏ) → Cosine ≈ 1 → Giống nhau
              <br />• Vuông góc → Cosine ≈ 0 → Không liên quan  
              <br />• Ngược hướng → Cosine ≈ -1 → Trái ngược
            </p>
          </div>
        </div>
      )
    },

    // Slide 6: Interactive Demo Enhanced
    {
      title: "Demo Tương Tác: Movie Similarity Calculator",
      content: (
        <div className="space-y-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-4">🎬 Movie Dataset với Vector [Action, Romance, Comedy]</h3>
            <div className="grid grid-cols-3 gap-2 text-xs">
              {movies.map((movie, i) => (
                <div key={i} className="bg-white p-2 rounded text-center">
                  <div className="font-bold">{movie.name}</div>
                  <div className="text-gray-500">[{movie.vector.join(', ')}]</div>
                  <div className="text-gray-400 text-xs">{movie.genre}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-bold mb-2">🎯 Your Query Vector:</label>
                <div className="flex space-x-2">
                  {demoQuery.map((val, i) => (
                    <input
                      key={i}
                      type="number"
                      value={val}
                      onChange={(e) => {
                        const newQuery = [...demoQuery];
                        newQuery[i] = parseFloat(e.target.value) || 0;
                        setDemoQuery(newQuery);
                      }}
                      className="w-16 p-2 border rounded text-center"
                      min="0"
                      max="10"
                    />
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-1">[Action, Romance, Comedy]</p>
                <p className="text-xs text-blue-600 mt-1">💡 Thử: [2,8,7] = romantic musical, [9,1,2] = pure action</p>
              </div>

              <div>
                <label className="block text-sm font-bold mb-2">📊 Similarity Metric:</label>
                <select 
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  className="w-full p-2 border rounded"
                >
                  <option value="euclidean">Euclidean Distance (lower = more similar)</option>
                  <option value="cosine">Cosine Similarity (higher = more similar)</option>
                  <option value="both">Both (ranked by Euclidean)</option>
                </select>
              </div>

              <div className="bg-blue-50 p-3 rounded-lg">
                <h5 className="font-bold text-blue-700 mb-2">🧮 Live Calculation:</h5>
                <div className="text-xs font-mono space-y-1">
                  <p>Query: [{demoQuery.join(', ')}]</p>
                  <p>Top match: {getSortedResults()[0]?.name}</p>
                  <p>Distance: {getSortedResults()[0]?.euclidean.toFixed(2)}</p>
                  <p>Cosine: {getSortedResults()[0]?.cosine.toFixed(3)}</p>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-bold mb-2">🏆 Results (Ranked by Similarity):</h4>
              <div className="space-y-2">
                {getSortedResults().map((movie, i) => (
                  <div key={i} className={`p-3 rounded-lg ${i === 0 ? 'bg-green-100 border-2 border-green-400 shadow-md' : 'bg-gray-100'}`}>
                    <div className="flex justify-between items-center">
                      <div>
                        <span className="font-bold">#{i + 1} {movie.name}</span>
                        {i === 0 && <span className="ml-2 text-green-600 text-sm">👑 Best Match!</span>}
                      </div>
                      <div className="text-xs text-gray-600">
                        {selectedMetric !== 'cosine' && <span>Euclidean: {movie.euclidean.toFixed(2)} </span>}
                        {selectedMetric !== 'euclidean' && <span>Cosine: {movie.cosine.toFixed(3)}</span>}
                      </div>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">{movie.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg">
            <h4 className="font-bold text-purple-700 mb-2">🧮 Vector Arithmetic Magic:</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p><strong>Movie Math:</strong> Action + Romance = ?</p>
                <p className="font-mono text-xs">[9,2,3] + [2,8,7] = [11,10,10] ≈ action-romance epic</p>
              </div>
              <div>
                <p><strong>Famous Example:</strong> king - man + woman = queen</p>
                <p className="text-xs text-gray-600">Vector space captures relationships!</p>
              </div>
            </div>
          </div>
        </div>
      )
    },

    // Slide 7: Map Search Example
    {
      title: "Ứng dụng thực tế: Tìm kiếm vị trí trên bản đồ",
      content: (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-green-50 to-blue-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-center mb-4">🗺️ GPS Thông minh với Vector Search</h3>
            <p className="text-center text-gray-600">Không cần địa chỉ chính xác - chỉ cần mô tả ý định!</p>
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="space-y-4">
              <div className="bg-red-50 p-4 rounded-lg border-l-4 border-red-400">
                <h4 className="font-bold text-red-700 mb-3">❌ Traditional GPS Search</h4>
                <div className="space-y-2 text-sm">
                  <div className="bg-white p-2 rounded">
                    <p><strong>Query:</strong> "Coffee near park"</p>
                    <p className="text-red-600">Result: No exact address match found</p>
                  </div>
                  <div className="bg-white p-2 rounded">
                    <p><strong>Query:</strong> "Quán cà phê gần công viên"</p>
                    <p className="text-red-600">Result: Please enter exact address</p>
                  </div>
                </div>
                <p className="text-xs text-gray-600 mt-2">Cần address chính xác: "123 Nguyễn Huệ, Q1"</p>
              </div>

              <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-400">
                <h4 className="font-bold text-green-700 mb-3">✅ Vector-powered GPS</h4>
                <div className="space-y-2 text-sm">
                  <div className="bg-white p-2 rounded">
                    <p><strong>Query:</strong> "Coffee near park"</p>
                    <p className="text-green-600">→ Finds cafés with outdoor seating near green spaces</p>
                  </div>
                  <div className="bg-white p-2 rounded">
                    <p><strong>Query:</strong> "Chỗ làm việc yên tĩnh có wifi"</p>
                    <p className="text-green-600">→ Finds coworking spaces, quiet cafés with good internet</p>
                  </div>
                </div>
                <p className="text-xs text-gray-600 mt-2">Hiểu intent & context, không cần exact keywords</p>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-bold text-gray-800 mb-2">📍 Location Vector Example:</h4>
              <div className="bg-white p-4 rounded-lg border">
                <div className="space-y-3 text-sm">
                  {locations.map((loc, i) => (
                    <div key={i} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                      <div>
                        <p className="font-semibold">{loc.name}</p>
                        <p className="text-gray-500">{loc.type}</p>
                      </div>
                      <div className="text-xs font-mono">
                        [{loc.vector.join(',')}]
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-blue-50 p-4 rounded-lg">
                <h5 className="font-bold text-blue-700 mb-2">🎯 Vector Dimensions:</h5>
                <div className="text-sm space-y-1">
                  <p><strong>[Ambience, Service, Convenience]</strong></p>
                  <p>• Ambience: 1-10 (cozy → formal)</p>
                  <p>• Service: 1-10 (self → full service)</p>
                  <p>• Convenience: 1-10 (hidden gem → accessible)</p>
                </div>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg">
                <h5 className="font-bold text-yellow-700 mb-2">🔍 Example Search:</h5>
                <p className="text-sm">
                  Query: "cozy coffee place" → Vector [8, 4, 6]
                  <br />→ Matches Highlands Coffee [7, 4, 8]
                  <br />→ High ambience, moderate service, good convenience
                </p>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 p-6 rounded-lg">
            <h4 className="font-bold text-purple-700 mb-4">🌟 Real-world Implementation:</h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <MapPin className="w-8 h-8 mx-auto text-purple-600 mb-2" />
                <p><strong>Google Maps</strong></p>
                <p className="text-gray-600">"Restaurants near me" understands context</p>
              </div>
              <div className="text-center">
                <Coffee className="w-8 h-8 mx-auto text-purple-600 mb-2" />
                <p><strong>Foursquare</strong></p>
                <p className="text-gray-600">Venue recommendations by vibe & mood</p>
              </div>
              <div className="text-center">
                <ShoppingCart className="w-8 h-8 mx-auto text-purple-600 mb-2" />
                <p><strong>Uber Eats</strong></p>
                <p className="text-gray-600">"Healthy dinner options" semantic search</p>
              </div>
            </div>
          </div>
        </div>
      )
    },

    // Slide 8: Indexing Deep Dive
    {
      title: "Indexing: Làm thế nào tìm được nhanh trong millions vectors?",
      content: (
        <div className="space-y-6">
          <div className="bg-red-50 p-4 rounded-lg border-l-4 border-red-400">
            <h3 className="font-bold text-red-700 mb-2">⚠️ Problem: Linear Search không scale</h3>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <p><strong>1M vectors:</strong></p>
                <p className="text-red-600">~1 second</p>
              </div>
              <div className="text-center">
                <p><strong>10M vectors:</strong></p>
                <p className="text-red-600">~10 seconds</p>
              </div>
              <div className="text-center">
                <p><strong>100M vectors:</strong></p>
                <p className="text-red-600">~100 seconds 💀</p>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 p-6 rounded-lg">
            <h3 className="font-bold text-blue-700 mb-4">🚀 Solution: ANN (Approximate Nearest Neighbor) Indexing</h3>
            <p className="text-gray-700 mb-4">Trade-off nhỏ về accuracy để có speed khủng khiếp</p>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-bold text-blue-700 mb-3 flex items-center">
                <Layers className="w-5 h-5 mr-2" />
                HNSW
              </h4>
              <div className="text-sm space-y-2">
                <p><strong>Hierarchical Navigable Small World</strong></p>
                <div className="bg-blue-50 p-2 rounded">
                  <p className="font-semibold">Cách hoạt động:</p>
                  <ul className="text-xs space-y-1 mt-1">
                    <li>• Multi-layer graph</li>
                    <li>• Layer cao: long-range connections</li>
                    <li>• Layer thấp: local connections</li>
                    <li>• Search: top → bottom</li>
                  </ul>
                </div>
                <div className="bg-green-50 p-2 rounded text-xs">
                  <p><strong>Ưu điểm:</strong> Real-time, high accuracy</p>
                  <p><strong>Nhược điểm:</strong> Memory intensive</p>
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-bold text-green-700 mb-3 flex items-center">
                <Grid3x3 className="w-5 h-5 mr-2" />
                IVF
              </h4>
              <div className="text-sm space-y-2">
                <p><strong>Inverted File Index</strong></p>
                <div className="bg-green-50 p-2 rounded">
                  <p className="font-semibold">Cách hoạt động:</p>
                  <ul className="text-xs space-y-1 mt-1">
                    <li>• Chia vectors thành clusters</li>
                    <li>• Search chỉ trong nearest clusters</li>
                    <li>• Skip xa clusters</li>
                    <li>• Tunable speed vs accuracy</li>
                  </ul>
                </div>
                <div className="bg-green-50 p-2 rounded text-xs">
                  <p><strong>Ưu điểm:</strong> Scalable, configurable</p>
                  <p><strong>Nhược điểm:</strong> Cluster quality matters</p>
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg border">
              <h4 className="font-bold text-purple-700 mb-3 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                PQ
              </h4>
              <div className="text-sm space-y-2">
                <p><strong>Product Quantization</strong></p>
                <div className="bg-purple-50 p-2 rounded">
                  <p className="font-semibold">Cách hoạt động:</p>
                  <ul className="text-xs space-y-1 mt-1">
                    <li>• Compress vectors</li>
                    <li>• Sub-vector quantization</li>
                    <li>• Lookup table for fast distance</li>
                    <li>• Huge memory savings</li>
                  </ul>
                </div>
                <div className="bg-purple-50 p-2 rounded text-xs">
                  <p><strong>Ưu điểm:</strong> Memory efficient</p>
                  <p><strong>Nhược điểm:</strong> Some accuracy loss</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg">
            <h3 className="font-bold text-gray-800 mb-4">📚 Phân biệt quan trọng:</h3>
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-bold text-blue-600 mb-2">📇 INDEX</h4>
                <ul className="text-sm space-y-1">
                  <li>• Cấu trúc dữ liệu tăng tốc</li>
                  <li>• Không thêm ý nghĩa</li>
                  <li>• HNSW, IVF, B-Tree...</li>
                  <li>• Like mục lục sách</li>
                </ul>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-bold text-green-600 mb-2">🏷️ LABEL/CATEGORY</h4>
                <ul className="text-sm space-y-1">
                  <li>• Tags do người gán</li>
                  <li>• Có semantic meaning</li>
                  <li>• "action", "romance"...</li>
                  <li>• For filtering & training</li>
                </ul>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <h4 className="font-bold text-purple-600 mb-2">🎯 CLUSTER</h4>
                <ul className="text-sm space-y-1">
                  <li>• Nhóm do máy phát hiện</li>
                  <li>• Dựa trên similarity</li>
                  <li>• K-means, DBSCAN...</li>
                  <li>• Unsupervised grouping</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 p-4 rounded-lg">
            <h4 className="font-bold text-yellow-700 mb-2">⚡ Performance Comparison:</h4>
            <div className="grid grid-cols-2 gap-6 text-sm">
              <div>
                <p><strong>Linear Search (100M vectors):</strong></p>
                <p className="text-red-600">~100 seconds, 100% accuracy</p>
              </div>
              <div>
                <p><strong>HNSW Index (100M vectors):</strong></p>
                <p className="text-green-600">~20ms, 95%+ accuracy</p>
              </div>
            </div>
            <p className="text-xs text-gray-600 mt-2">5000x faster với minimal accuracy loss!</p>
          </div>
        </div>
      )
    },

    // Slide 9: RAG Deep Dive
    {
      title: "RAG: Retrieval-Augmented Generation",
      content: (
        <div className="space-y-6">
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-lg">
            <h3 className="text-xl font-bold text-center mb-4">🧠 RAG = LLM (ChatGPT) + VectorDB (Smart Memory)</h3>
            <p className="text-center text-gray-600">Kết hợp khả năng nói chuyện thông minh với thông tin chính xác từ dữ liệu của bạn</p>
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="space-y-4">
              <div className="bg-white p-4 rounded-lg border">
                <h4 className="font-bold text-blue-600 mb-3">🤖 LLM Component</h4>
                <ul className="text-sm space-y-2">
                  <li>• <strong>ChatGPT, Claude, Gemini...</strong></li>
                  <li>• Rất giỏi viết, trả lời tự nhiên</li>
                  <li>• Hiểu ngữ cảnh, reasoning</li>
                  <li>• <span className="text-red-600">Nhược điểm:</span> Có thể hallucinate, outdated knowledge</li>
                </ul>
              </div>

              <div className="bg-white p-4 rounded-lg border">
                <h4 className="font-bold text-green-600 mb-3">🗃️ VectorDB Component</h4>
                <ul className="text-sm space-y-2">
                  <li>• <strong>Pinecone, Weaviate, Chroma...</strong></li>
                  <li>• Lưu embedding của documents</li>
                  <li>• Rất giỏi tìm đoạn văn liên quan</li>
                  <li>• <span className="text-green-600">Ưu điểm:</span> Accurate, up-to-date, traceable</li>
                </ul>
              </div>

              <div className="bg-yellow-50 p-4 rounded-lg border-l-4 border-yellow-400">
                <h4 className="font-bold text-yellow-700 mb-2">💡 RAG = Best of Both Worlds</h4>
                <p className="text-sm text-gray-700">
                  LLM cung cấp intelligence & natural language, VectorDB cung cấp accurate & relevant information
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="font-bold text-gray-800 mb-2">🔄 RAG Workflow:</h4>
              <div className="space-y-3">
                <div className="bg-blue-50 p-3 rounded flex items-center">
                  <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">1</div>
                  <div>
                    <p className="font-semibold">Prepare Data</p>
                    <p className="text-xs text-gray-600">PDF, docs, wiki → chunks → embeddings → VectorDB</p>
                  </div>
                </div>

                <div className="bg-green-50 p-3 rounded flex items-center">
                  <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">2</div>
                  <div>
                    <p className="font-semibold">User Question</p>
                    <p className="text-xs text-gray-600">"Nghỉ phép 3 ngày được không?" → embedding</p>
                  </div>
                </div>

                <div className="bg-purple-50 p-3 rounded flex items-center">
                  <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">3</div>
                  <div>
                    <p className="font-semibold">Vector Retrieval</p>
                    <p className="text-xs text-gray-600">Find top-k relevant document chunks</p>
                  </div>
                </div>

                <div className="bg-orange-50 p-3 rounded flex items-center">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">4</div>
                  <div>
                    <p className="font-semibold">LLM Generation</p>
                    <p className="text-xs text-gray-600">Question + Context → Natural answer với citations</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-green-50 p-6 rounded-lg border-l-4 border-green-400">
            <h4 className="font-bold text-green-700 mb-4">✨ RAG Example: Company Chatbot</h4>
            
            <div className="bg-white p-4 rounded-lg mb-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p><strong>🙋 User:</strong> "Công ty có chính sách nghỉ thai sản như thế nào?"</p>
                </div>
                <div>
                  <p><strong>🔍 VectorDB finds:</strong> HR policy section 4.2 về maternal leave</p>
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-lg">
              <p><strong>🤖 RAG Response:</strong></p>
              <div className="text-sm mt-2 bg-gray-50 p-3 rounded">
                <p className="italic">
                  "Theo quy định công ty, nhân viên nữ được nghỉ thai sản 6 tháng với lương cơ bản 100%. 
                  Bạn cần nộp đơn trước ít nhất 1 tháng và kèm theo giấy xác nhận từ bác sĩ."
                </p>
                <p className="text-blue-600 mt-2 text-xs">
                  [Nguồn: Employee Handbook v2.3, Section 4.2, Page 15]
                </p>
              </div>
            </div>

            <p className="text-xs text-green-600 mt-2">⚡ Response time: &lt;2 seconds, Accuracy: 95%+, With source citation</p>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg">
            <h4 className="font-bold text-purple-700 mb-2">🔗 MCP (Model Context Protocol) trong RAG</h4>
            <p className="text-sm text-gray-700">
              <strong>MCP</strong> là protocol để LLM gọi external tools (như VectorDB) trong real-time:
              <br />• LLM: "Tôi cần info về leave policy..."
              <br />• MCP: Kết nối với VectorDB → retrieve relevant docs
              <br />• LLM: Nhận context → generate informed response
              <br /><br />
              💡 <strong>Result:</strong> LLM không cần biết trước tất cả thông tin, có thể "learn" động khi trả lời!
            </p>
          </div>
        </div>
      )
    }
  ];

  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'ArrowRight') nextSlide();
      if (e.key === 'ArrowLeft') prevSlide();
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [nextSlide, prevSlide]);

  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col">
      <div className="bg-white shadow-sm border-b flex-shrink-0">
        <div className="max-w-6xl mx-auto px-4 py-3 flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <button 
              onClick={prevSlide}
              className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
              disabled={currentSlide === 0}
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            <span className="text-sm font-medium text-gray-600">
              {currentSlide + 1} / {slides.length}
            </span>
            <button 
              onClick={nextSlide}
              className="p-2 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
              disabled={currentSlide === slides.length - 1}
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>
          
          <h2 className="text-lg font-bold text-gray-800">
            {slides[currentSlide].title}
          </h2>
          
          <div className="flex space-x-1">
            {slides.map((_, i) => (
              <button
                key={i}
                onClick={() => setCurrentSlide(i)}
                className={`w-2 h-2 rounded-full transition-colors ${
                  i === currentSlide ? 'bg-blue-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 max-w-6xl mx-auto px-4 py-8 w-full">
        <div className="bg-white rounded-lg shadow-lg p-8 h-full flex flex-col">
          <h1 className="text-2xl font-bold text-gray-800 mb-6 flex-shrink-0">
            {slides[currentSlide].title}
          </h1>
          
          <div className="flex-1 overflow-auto">
            {slides[currentSlide].content}
          </div>
        </div>
      </div>

      <div className="bg-white border-t py-4 flex-shrink-0">
        <div className="max-w-6xl mx-auto px-4 text-center">
          <div className="flex justify-center space-x-6 text-sm text-gray-500">
            <span>🎯 Vector Database Academic Deep Dive</span>
            <span>💡 Use arrow keys to navigate</span>
            <span>🚀 From theory to implementation</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VectorDBPresentation;