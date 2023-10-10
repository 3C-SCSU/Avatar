/* Authors: Brendan Chermack, Abby Sininger, and Eion Ness */
const { app, BrowserWindow } = require('electron');
const createWindow = () => {
	const win = new BrowserWindow({
		width: 1280,
		height: 720,
	});
	win.loadFile('index.html');
};
app.whenReady().then(() => {
	createWindow();
	app.on('activate', () => {
		if (BrowserWindow.getAllWindows().length === 0) {
			createWindow();
		}
	});
});
app.on('window-all-closed', () => {
	if (process.platform !== 'darwin') {
		app.quit();
	}
});
