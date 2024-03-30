const express            = require('express');

const port   = process.env.PORT||9000;
const host   = process.env.HOST||'0.0.0.0'||'127.0.0.1';
const server = express();

server.use(express.static('./app'));
server.use(express.json());
server.use(express.urlencoded({ extended: false }));

server.use("/@tensorflow"       , express.static('./node_modules/@tensorflow')); 

//assuming app is express Object.
server.get('/',function(req,res) {
    res.render('index.html');
  });  

server.listen(port, host, () => {
  console.log(`Server is running on http://${host}:${port}`);
});
