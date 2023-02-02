module.exports = {
  mode: 'development',
  entry: {
    main: './src/main.ts',
    worker: './src/worker.ts'
  },

  output: {
    filename: 'wgpy-[name].js',
    path: __dirname + '/dist',
    library: {
      name: 'wgpy',
      type: 'var',
    },
  },

  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
};
