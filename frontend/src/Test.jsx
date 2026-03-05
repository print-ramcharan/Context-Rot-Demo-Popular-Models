export default function Test() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center">
      
      <div className="bg-white p-10 rounded-2xl shadow-2xl text-center space-y-4">
        
        <h1 className="text-4xl font-bold text-gray-800">
          Tailwind is Worrrrking 🎉
        </h1>

        <p className="text-gray-600">
          If you see colors, spacing, and shadows — you're good.
        </p>

        <button className="px-6 py-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition duration-300">
          Hover Me
        </button>

        <div className="mt-6 grid grid-cols-3 gap-3">
          <div className="h-12 bg-red-400 rounded"></div>
          <div className="h-12 bg-green-400 rounded"></div>
          <div className="h-12 bg-blue-400 rounded"></div>
        </div>

      </div>

    </div>
  );
}
