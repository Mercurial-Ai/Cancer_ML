
const { app, BrowserWindow, Menu, shell } = require('electron')
const path = require('path')

function createWindow () {
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })

  mainWindow.loadFile('index.html')

}

const template = [
  {
    label: 'Cancer ML',
    submenu: [
      {
        label: 'About',
        click() {
          shell.openExternal("https://github.com/Tpool1/Cancer_ML")
        }
      },
      {
        label: 'Exit',
        click() {
          app.quit()
        }
      }
    ]
  }
]

// adjust menu bar
var menu = Menu.buildFromTemplate(template)
Menu.setApplicationMenu(menu)

app.whenReady().then(() => {
  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})
