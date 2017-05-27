//
// Created by ed_grolsh on 27.05.17.
//

#include "get_face_points.hpp"

void get_face_points() {
//каждое лицо в этой проге берут в прямоугольник и находят на нем 68 точек
    //это нужно в констурктор----
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize(argv[1]) >> sp;
    std::vector<cv::Point> face_points;//вектор точек
    dlib::cv_image<bgr_pixel> cvImage(frame);
    std::vector<rectangle> dets = detector(cvImage);
    std::vector<full_object_detection> shapes;
    //------

    cvImage = frame;
    dets = detector(cvImage);
    //опеределяем прямоугольники для каждого лица
    // Now we will go ask the shape_predictor to tell us the pose of
    // each face we detected.
    for (unsigned long j = 0; j < dets.size(); ++j)//для всех обнаруженных лиц
    {
        full_object_detection shape = sp(cvImage, dets[j]);//здесь получаем объект из 68 точек на лице
        shapes.push_back(shape);
    }

    short int left_eye = 37;
    short int right_eye = 43;
    short int nose = 30;

    cv::Point temp;
    temp.x = shapes[0].part(left_eye).x();
    temp.y = shapes[0].part(left_eye).y();
    face_points.push_back(shapes[0].part(temp));

    temp.x = shapes[0].part(right_eye).x();
    temp.y = shapes[0].part(right_eye).y();
    face_points.push_back(shapes[0].part(temp));

    temp.x = shapes[0].part(nose).x();
    temp.y = shapes[0].part(nose).y();
    face_points.push_back(shapes[0].part(temp));
}
