#version 150
    in vec3 position;
    in vec2 inTexCoords;
    
    out vec3 newColor;
    out vec2 outTexCoords;
    void main()
    {
        gl_Position = vec4(position, 1.0f);
        outTexCoords = vec2(inTexCoords.x, 1.0f - inTexCoords.y);
    }