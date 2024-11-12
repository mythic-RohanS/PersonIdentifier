// Step 1: Create an array to hold labeled face descriptors
let labeledFaceDescriptors = [];

// Step 2: Add a function to load images and get descriptors
const loadLabeledFaceDescriptors = async () => {
    const labels = ["Michael Jordan", "Rohan","Aarav","Vijay","Apramay"]; // List of names corresponding to your reference images
    const imageUrls = [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Michael_Jordan_in_2014.jpg/220px-Michael_Jordan_in_2014.jpg',
        'rohan.jpg','Aarav.jpg','Vijay.jpg','Apramay.jpg'
    ];

    for (let i = 0; i < labels.length; i++) {
        const img = await faceapi.fetchImage(imageUrls[i]);
        const faceDescription = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        
        if (!faceDescription) {
            console.warn(`No face detected in the image for ${labels[i]}`);
            continue;
        }
        
        const labeledFaceDescriptor = new faceapi.LabeledFaceDescriptors(labels[i], [faceDescription.descriptor]);
        labeledFaceDescriptors.push(labeledFaceDescriptor);
    }
};

// Step 3: Update the `run` function to load descriptors and set up FaceMatcher
const run = async () => {
    // Load models and start the video stream
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const videoFeedEl = document.getElementById('video-feed');
    videoFeedEl.srcObject = stream;

    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
        faceapi.nets.faceExpressionNet.loadFromUri('./models'),
    ]);

    await loadLabeledFaceDescriptors();

    // Create the face matcher with the labeled descriptors
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

    // Canvas setup for displaying face data
    const canvas = document.getElementById('canvas');
    canvas.style.left = videoFeedEl.offsetLeft;
    canvas.style.top = videoFeedEl.offsetTop;
    canvas.height = videoFeedEl.height;
    canvas.width = videoFeedEl.width;

    setInterval(async () => {
        const faceAIData = await faceapi.detectAllFaces(videoFeedEl).withFaceLandmarks().withFaceDescriptors().withAgeAndGender().withFaceExpressions();

        // Clear the canvas and resize results to match video dimensions
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        const resizedResults = faceapi.resizeResults(faceAIData, videoFeedEl);
        
        faceapi.draw.drawDetections(canvas, resizedResults);
        faceapi.draw.drawFaceLandmarks(canvas, resizedResults);
        faceapi.draw.drawFaceExpressions(canvas, resizedResults);

        resizedResults.forEach(face => {
            const { age, gender, genderProbability, detection, descriptor } = face;
            const genderText = `${gender} - ${Math.round(genderProbability * 100)}%`;
            const ageText = `${Math.round(age)} years`;
            const textField = new faceapi.draw.DrawTextField([genderText, ageText], face.detection.box.topRight);
            textField.draw(canvas);

            // Match against the faceMatcher
            const bestMatch = faceMatcher.findBestMatch(descriptor);
            const options = { label: bestMatch.toString() };
            const drawBox = new faceapi.draw.DrawBox(detection.box, options);
            drawBox.draw(canvas);
        });
    }, 200);
};

run();
