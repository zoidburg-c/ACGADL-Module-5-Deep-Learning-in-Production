const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require(path.resolve(__dirname, '../../.claude-office-skills/html2pptx-local.cjs'));

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.author = 'SMU Academy';
    pptx.title = 'Module 5 Session 4 - Deploying Models On-premise';

    const slidesDir = __dirname;
    const slideFiles = [];
    for (let i = 1; i <= 12; i++) {
        slideFiles.push(`slide-${String(i).padStart(2, '0')}.html`);
    }

    for (const file of slideFiles) {
        const htmlPath = path.join(slidesDir, file);
        console.log(`Processing ${file}...`);
        await html2pptx(htmlPath, pptx);
    }

    const outputPath = path.join(slidesDir, '..', 'Session 4 - Deploying Models On-premise.pptx');
    await pptx.writeFile({ fileName: outputPath });
    console.log(`Presentation saved to: ${outputPath}`);
}

createPresentation().catch(err => {
    console.error('Error creating presentation:', err.message);
    process.exit(1);
});
