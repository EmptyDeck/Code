Sub CreatePresentation()
    Dim objPresentation As Presentation
    Dim objSlide As Slide
    Dim objShape As Shape
    Dim objTextBox As Shape
    Dim SlideCount As Integer

    Set objPresentation = ActivePresentation

    For SlideCount = 1 To 7
        Set objSlide = objPresentation.Slides.Add(SlideCount, ppLayoutBlank)
        Select Case SlideCount
            Case 1
                Set objShape = objSlide.Shapes.AddShape(msoShapeRectangle, 10, 50, 710, 340)
                objShape.Fill.ForeColor.RGB = RGB(190, 190, 255)
                objShape.Line.ForeColor.RGB = RGB(0, 0, 255)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 60, 700, 100)
                objTextBox.TextFrame.TextRange.Text = "FK5103 DC ON 딜레이 타이머"
                objTextBox.TextFrame.TextRange.Font.Size = 36
                objTextBox.TextFrame.TextRange.Font.Bold = True
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 255)
                
                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 170, 700, 200)
                objTextBox.TextFrame.TextRange.Text = "1. 소개" & vbCrLf & "2. FK5103이란?" & vbCrLf & "3. 주요 기능" & vbCrLf & "4. 응용 분야" & vbCrLf & "5. 설치 및 연결" & vbCrLf & "6. 사용 및 설정" & vbCrLf & "7. 안전 주의사항"
                objTextBox.TextFrame.TextRange.Font.Size = 24
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 128)
            Case 2
                Set objShape = objSlide.Shapes.AddShape(msoShapeRectangle, 10, 50, 710, 100)
                objShape.Fill.ForeColor.RGB = RGB(0, 128, 0)
                objShape.Line.ForeColor.RGB = RGB(0, 64, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 60, 700, 100)
                objTextBox.TextFrame.TextRange.Text = "FK5103이란?"
                objTextBox.TextFrame.TextRange.Font.Size = 32
                objTextBox.TextFrame.TextRange.Font.Bold = True
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)

                Set objShape = objSlide.Shapes.AddShape(msoShapeRoundedRectangle, 10, 170, 710, 220)
                objShape.Fill.ForeColor.RGB = RGB(235, 235, 235)
                objShape.Line.ForeColor.RGB = RGB(0, 128, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 180, 700, 200)
                objTextBox.TextFrame.TextRange.Text = "FK5103은 DC 전원을 켜고 끄는 시간을 제어하는 타이머입니다."
                objTextBox.TextFrame.TextRange.Font.Size = 24
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(0, 128, 0)

            Case 3
                Set objShape = objSlide.Shapes.AddShape(msoShapeRectangle, 10, 50, 710, 100)
                objShape.Fill.ForeColor.RGB = RGB(255, 0, 0)
                objShape.Line.ForeColor.RGB = RGB(128, 0, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 60, 700, 100)
                objTextBox.TextFrame.TextRange.Text = "주요 기능"
                objTextBox.TextFrame.TextRange.Font.Size = 32
                objTextBox.TextFrame.TextRange.Font.Bold = True
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)

                Set objShape = objSlide.Shapes.AddShape(msoShapeRoundedRectangle, 10, 170, 710, 220)
                objShape.Fill.ForeColor.RGB = RGB(235, 235, 235)
                objShape.Line.ForeColor.RGB = RGB(255, 0, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 180, 700, 200)
                objTextBox.TextFrame.TextRange.Text = "1. 정밀한 시간 조절 기능" & vbCrLf & "2. 사용자 정의 시간 범위" & vbCrLf & "3. 내구성 및 신뢰성"
                objTextBox.TextFrame.TextRange.Font.Size = 24
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 0, 0)

            Case 4
                Set objShape = objSlide.Shapes.AddShape(msoShapeRectangle, 10, 50, 710, 100)
                objShape.Fill.ForeColor.RGB = RGB(0, 0, 255)
                objShape.Line.ForeColor.RGB = RGB(0, 0, 128)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 60, 700, 100)
                objTextBox.TextFrame.TextRange.Text = "응용 분야"
                objTextBox.TextFrame.TextRange.Font.Size = 32
                objTextBox.TextFrame.TextRange.Font.Bold = True
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)

                Set objShape = objSlide.Shapes.AddShape(msoShapeRoundedRectangle, 10, 170, 710, 220)
                objShape.Fill.ForeColor.RGB = RGB(235, 235, 235)
                objShape.Line.ForeColor.RGB = RGB(0, 0, 255)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 180, 700, 200)
                objTextBox.TextFrame.TextRange.Text = "1. 산업용 장비" & vbCrLf & "2. 자동차 제어" & vbCrLf & "3. 전력 시스템 관리"
                objTextBox.TextFrame.TextRange.Font.Size = 24
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(0, 0, 255)

            Case 5
                Set objShape = objSlide.Shapes.AddShape(msoShapeRectangle, 10, 50,710, 100)
                objShape.Fill.ForeColor.RGB = RGB(128, 0, 128)
                objShape.Line.ForeColor.RGB = RGB(64, 0, 64)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 60, 700, 100)
                objTextBox.TextFrame.TextRange.Text = "설치 및 연결"
                objTextBox.TextFrame.TextRange.Font.Size = 32
                objTextBox.TextFrame.TextRange.Font.Bold = True
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)

                Set objShape = objSlide.Shapes.AddShape(msoShapeRoundedRectangle, 10, 170, 710, 220)
                objShape.Fill.ForeColor.RGB = RGB(235, 235, 235)
                objShape.Line.ForeColor.RGB = RGB(128, 0, 128)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 180, 700, 200)
                objTextBox.TextFrame.TextRange.Text = "1. 전원 공급 장치에 타이머 연결" & vbCrLf & "2. 타이머의 입력 단자와 출력 단자 연결" & vbCrLf & "3. 올바른 전압 및 전류 값으로 설정"
                objTextBox.TextFrame.TextRange.Font.Size = 24
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(128, 0, 128)

            Case 6
                Set objShape = objSlide.Shapes.AddShape(msoShapeRectangle, 10, 50, 710, 100)
                objShape.Fill.ForeColor.RGB = RGB(0, 128, 0)
                objShape.Line.ForeColor.RGB = RGB(0, 64, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 60, 700, 100)
                objTextBox.TextFrame.TextRange.Text = "사용 및 설정"
                objTextBox.TextFrame.TextRange.Font.Size = 32
                objTextBox.TextFrame.TextRange.Font.Bold = True
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)

                Set objShape = objSlide.Shapes.AddShape(msoShapeRoundedRectangle, 10, 170, 710, 220)
                objShape.Fill.ForeColor.RGB = RGB(235, 235, 235)
                objShape.Line.ForeColor.RGB = RGB(0, 128, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 180, 700, 200)
                objTextBox.TextFrame.TextRange.Text = "1. 딜레이 시간 설정" & vbCrLf & "2. 작동 모드 선택" & vbCrLf & "3. 필요한 경우 부가 기능 사용"
                objTextBox.TextFrame.TextRange.Font.Size = 24
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(0, 128, 0)

            Case 7
                Set objShape = objSlide.Shapes.AddShape(msoShapeRectangle, 10, 50, 710, 100)
                objShape.Fill.ForeColor.RGB = RGB(255, 0, 0)
                objShape.Line.ForeColor.RGB = RGB(128, 0, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 60, 700, 100)
                objTextBox.TextFrame.TextRange.Text = "안전 주의사항"
                objTextBox.TextFrame.TextRange.Font.Size = 32
                objTextBox.TextFrame.TextRange.Font.Bold = True
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)

                Set objShape = objSlide.Shapes.AddShape(msoShapeRoundedRectangle, 10, 170, 710, 220)
                objShape.Fill.ForeColor.RGB = RGB(235, 235, 235)
                objShape.Line.ForeColor.RGB = RGB(255, 0, 0)
                objShape.Line.Weight = 2

                Set objTextBox = objSlide.Shapes.AddTextbox(msoTextOrientationHorizontal, 20, 180, 700, 200)
                objTextBox.TextFrame.TextRange.Text = "1. 전원을 끄고 작업하기 전에 완전히 분리하세요." & vbCrLf & "2. 전문가의 지시에 따라 설치 및 유지 관리를 수행하세요." & vbCrLf & "3. 과부하 방지를 위해 올바른 전류 및 전압 값을 사용하세요."
                objTextBox.TextFrame.TextRange.Font.Size = 24
                objTextBox.TextFrame.TextRange.Font.Color.RGB = RGB(255, 0, 0)
    End Select
Next SlideCount
End Sub


