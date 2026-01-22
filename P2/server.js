import express from 'express'

const app = express()
app.use(express.json())


/*Get - listar
Post - Criar
Put - Editar vários
Patch - Editar Um
Delete - Deletar */

app.post('/usuarios', async (req, res) => {

    await prisma.user.create({
        data:{
            email: req.body.email,
            name: req.body.name,
            age: req.body.age
        }
    })
    
    res.status(201).json(req.body)
    
})

app.get('/usuarios', async (req, res) => {

    const users = await prisma.user.findMany()

    res.status(200).json(users)

})

app.listen(3000)

